// Benchmark driver: CPU reference vs GPU variants, with recall, launch shape and throughput.
//
//   veckern_bench --n 1000000 --d 128 --m 16 --nq 1000 --k 10                       (synthetic data)
//   veckern_bench --base sift_base.fvecs --query sift_query.fvecs --gt sift_groundtruth.ivecs --m 16
//   veckern_bench --n 10000000 --variants v4,v6,v7 --queries-per-block 4 --label sweep   (subset + tuning)
#include <algorithm>
#include <cstdio>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "veckern/cpu_ref.hpp"
#include "veckern/io.hpp"
#ifdef VECKERN_HAS_CUDA
#include "veckern/gpu.hpp"
#endif

using namespace veckern;

namespace {

struct Args {
    std::size_t n = 1'000'000, d = 128, m = 16, nq = 1000, k = 10, clusters = 100, seed = 1, latent_dim = 16;
    std::size_t train = 100'000, iters = 15, reps = 5, eval_queries = 100, cpu_queries = 100, rerank = 0;
    std::size_t threads = 256, blocks_per_query = 0, queries_per_block = 2, ilp = 4;
    std::string base, query, gt, csv = "results.csv", variants = "all", label = "default";
};

void usage() {
    std::puts(
        "data:     --n --d --clusters --latent-dim --seed | --base X.fvecs|X.bvecs --query X.fvecs|X.bvecs [--gt X.ivecs]\n"
        "search:   --m --nq --k --rerank R\n"
        "training: --train --iters\n"
        "measure:  --reps --eval-queries (0 = skip recall) --cpu-queries (0 = skip CPU)\n"
        "gpu:      --variants all|v0,v1,...,v8  --threads T  --blocks-per-query B  --queries-per-block Q  --ilp 1|2|4\n"
        "output:   --csv out.csv  --label name");
}

Args parse(int argc, char** argv) {
    Args a;
    std::map<std::string, std::size_t*> nums = {
        {"--n", &a.n}, {"--d", &a.d}, {"--m", &a.m}, {"--nq", &a.nq}, {"--k", &a.k},
        {"--clusters", &a.clusters}, {"--seed", &a.seed}, {"--latent-dim", &a.latent_dim}, {"--train", &a.train}, {"--iters", &a.iters},
        {"--reps", &a.reps}, {"--eval-queries", &a.eval_queries}, {"--cpu-queries", &a.cpu_queries},
        {"--rerank", &a.rerank}, {"--threads", &a.threads}, {"--blocks-per-query", &a.blocks_per_query},
        {"--queries-per-block", &a.queries_per_block}, {"--ilp", &a.ilp}};
    std::map<std::string, std::string*> strs = {{"--base", &a.base}, {"--query", &a.query}, {"--gt", &a.gt},
                                                {"--csv", &a.csv}, {"--variants", &a.variants},
                                                {"--label", &a.label}};
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        if (key == "-h" || key == "--help") {
            usage();
            std::exit(0);
        }
        check(i + 1 < argc, "missing value for " + key);
        if (nums.count(key)) *nums[key] = std::stoull(argv[++i]);
        else if (strs.count(key)) *strs[key] = argv[++i];
        else throw std::runtime_error("unknown option " + key + " (try --help)");
    }
    return a;
}

std::vector<float> load_vectors(const std::string& path, std::size_t& n, std::size_t& d, std::size_t max_n) {
    const bool bytes = path.size() > 6 && path.substr(path.size() - 6) == ".bvecs";
    return bytes ? read_bvecs(path, n, d, max_n) : read_fvecs(path, n, d, max_n);
}

double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

bool selected(const std::string& list, const std::string& variant_name) {
    if (list == "all") return true;
    std::stringstream ss(list);
    std::string item;
    while (std::getline(ss, item, ','))
        if (!item.empty() && variant_name.rfind(item, 0) == 0) return true;  // prefix match: "v6" or full name
    return false;
}

struct Row {
    std::string variant;
    int threads = 0, blocks_x = 0, grid_y = 0, qpb = 1, ilp = 1, blocks_per_sm = 0;
    double waves = 0, lut_ms = 0, scan_ms = 0, select_ms = 0, transfer_ms = 0, total_ms = 0, qps = 0, code_gbs = 0;
    double recall = -1;
};

}  // namespace

int main(int argc, char** argv) try {
    Args a = parse(argc, argv);
    check(a.k >= 1 && a.k <= 128, "--k must be in [1, 128]");

    // ---- data
    Timer data_timer;
    std::vector<float> base, queries;
    if (!a.base.empty()) {
        check(!a.query.empty(), "--base requires --query");
        base = load_vectors(a.base, a.n, a.d, a.n);
        std::size_t qd = 0;
        queries = load_vectors(a.query, a.nq, qd, a.nq);
        check(qd == a.d, "query and base dimensions differ");
    } else {
        // Queries are further rows of the same source: same clusters, never equal to a base row.
        SyntheticSource src(a.d, a.clusters, a.seed, a.latent_dim);
        base.resize(a.n * a.d);
        queries.resize(a.nq * a.d);
        src.fill(0, a.n, base.data());
        src.fill(std::size_t(1) << 40, a.nq, queries.data());
    }
    a.eval_queries = std::min(a.eval_queries, a.nq);
    a.cpu_queries = std::min(a.cpu_queries, a.nq);
    std::printf("data: n=%zu d=%zu nq=%zu (%.1f s) | PQ m=%zu (%zu B/vector, %.0fx compression) | k=%zu\n", a.n,
                a.d, a.nq, data_timer.ms() / 1e3, a.m, a.m, 4.0 * a.d / a.m, a.k);

    // ---- PQ training and encoding
    Timer train_timer;
    ProductQuantizer pq(a.d, a.m);
    pq.train(base.data(), std::min(a.train, a.n), a.iters);
    std::vector<uint8_t> codes(a.n * a.m);
    pq.encode(base.data(), a.n, codes.data());
    std::printf("PQ train+encode: %.1f s\n", train_timer.ms() / 1e3);

    // ---- ground truth for the first eval_queries queries
    std::vector<int32_t> truth;
    std::size_t truth_stride = a.k;
    if (a.eval_queries > 0) {
        if (!a.gt.empty()) {
            std::size_t gn = 0;
            truth = read_ivecs(a.gt, gn, truth_stride, a.nq);
            check(gn >= a.eval_queries && truth_stride >= a.k, "ground truth too small");
        } else {
            Timer gt_timer;
            std::vector<Neighbor> exact(a.eval_queries * a.k);
            exact_knn(base.data(), a.n, queries.data(), a.eval_queries, a.d, a.k, exact.data());
            truth.resize(exact.size());
            for (std::size_t j = 0; j < exact.size(); ++j) truth[j] = exact[j].id;
            std::printf("exact ground truth (%zu queries): %.1f s\n", a.eval_queries, gt_timer.ms() / 1e3);
        }
    }
    // With --rerank R the search returns R candidates and the CPU re-scores them exactly.
    const std::size_t k_search = a.rerank ? std::max(a.k, a.rerank) : a.k;
    auto recall = [&](const std::vector<Neighbor>& res) -> double {
        if (a.eval_queries == 0) return -1;
        if (a.rerank == 0) return recall_at_k(res.data(), k_search, truth.data(), truth_stride, a.eval_queries, a.k);
        std::vector<Neighbor> rr(a.eval_queries * a.k);
        rerank_exact(base.data(), a.d, queries.data(), a.eval_queries, res.data(), k_search, a.k, rr.data());
        return recall_at_k(rr.data(), a.k, truth.data(), truth_stride, a.eval_queries, a.k);
    };
    const double code_bytes = static_cast<double>(a.n) * a.m;  // logical bytes scanned per query

    std::vector<Row> rows;
    // ---- CPU reference (subset of queries; it is slow)
    if (a.cpu_queries > 0) {
        std::vector<Neighbor> res(a.cpu_queries * k_search);
        Timer t;
        adc_search(pq, codes.data(), a.n, queries.data(), a.cpu_queries, k_search, res.data());
        Row r;
        r.variant = "cpu_reference";
        r.total_ms = t.ms();
        r.qps = a.cpu_queries / (r.total_ms / 1e3);
        r.code_gbs = code_bytes * a.cpu_queries / (r.total_ms / 1e3) / 1e9;
        if (a.cpu_queries >= a.eval_queries) r.recall = recall(res);
        rows.push_back(r);
    }

#ifdef VECKERN_HAS_CUDA
    const auto info = gpu::device_info();
    std::printf("GPU: %s, %d SMs, %.0f GB/s theoretical DRAM peak\n", info.name.c_str(), info.sm_count,
                info.peak_bandwidth_gbs);
    gpu::PQSearcher searcher(pq, codes.data(), a.n);
    gpu::Tuning tuning;
    tuning.scan_threads = static_cast<int>(a.threads);
    tuning.blocks_per_query = static_cast<int>(a.blocks_per_query);
    tuning.queries_per_block = static_cast<int>(a.queries_per_block);
    tuning.vectors_per_step = static_cast<int>(a.ilp);
    searcher.set_tuning(tuning);
    using gpu::ScanVariant;
    for (auto v : {ScanVariant::GlobalLut, ScanVariant::SharedLut, ScanVariant::SharedLutVec, ScanVariant::Interleaved,
                   ScanVariant::Fused, ScanVariant::FusedReg, ScanVariant::FusedVec, ScanVariant::FusedVecMQ,
                   ScanVariant::FusedWarp}) {
        if (!selected(a.variants, gpu::variant_name(v))) continue;
        const bool needs16 =
            v == ScanVariant::SharedLutVec || v == ScanVariant::FusedVec || v == ScanVariant::FusedVecMQ ||
            v == ScanVariant::FusedWarp;
        if (needs16 && a.m % 16 != 0) continue;
        std::vector<Neighbor> res(a.nq * k_search);
        searcher.search(queries.data(), a.nq, k_search, v, res.data());  // warm-up (and layout build)
        std::vector<double> lut, scan, sel, xfer, total;
        for (std::size_t r = 0; r < a.reps; ++r) {
            const auto t = searcher.search(queries.data(), a.nq, k_search, v, res.data());
            lut.push_back(t.lut_ms);
            scan.push_back(t.scan_ms);
            sel.push_back(t.select_ms);
            xfer.push_back(t.transfer_ms);
            total.push_back(t.total_ms());
        }
        const auto shape = searcher.last_launch();
        Row row;
        row.variant = gpu::variant_name(v);
        row.threads = shape.threads;
        row.blocks_x = shape.blocks_x;
        row.grid_y = shape.grid_y;
        row.qpb = shape.queries_per_block;
        row.ilp = shape.vectors_per_step;
        row.blocks_per_sm = shape.blocks_per_sm;
        row.waves = shape.waves;
        row.lut_ms = median(lut);
        row.scan_ms = median(scan);
        row.select_ms = median(sel);
        row.transfer_ms = median(xfer);
        row.total_ms = median(total);
        row.qps = a.nq / (row.total_ms / 1e3);
        row.code_gbs = code_bytes * a.nq / (row.scan_ms / 1e3) / 1e9;
        res.resize(std::max<std::size_t>(a.eval_queries, 1) * k_search);
        row.recall = recall(res);
        rows.push_back(row);
    }
#else
    std::puts("(built without CUDA: GPU variants skipped)");
#endif

    // ---- report
    std::printf("\n%-18s %5s %5s %5s %2s %3s %5s %8s %8s %8s %8s %9s %10s %9s %7s\n", "variant", "T", "bx", "gy", "Q",
                "ILP", "waves", "lut ms", "scan ms", "sel ms", "xfer ms", "total ms", "QPS", "code GB/s", "recall");
    for (const auto& r : rows)
        std::printf("%-18s %5d %5d %5d %2d %3d %5.2f %8.2f %8.2f %8.2f %8.2f %9.2f %10.0f %9.1f %7.3f\n", r.variant.c_str(),
                    r.threads, r.blocks_x, r.grid_y, r.qpb, r.ilp, r.waves, r.lut_ms, r.scan_ms, r.select_ms,
                    r.transfer_ms, r.total_ms, r.qps, r.code_gbs, r.recall);
    std::puts("code GB/s = logical PQ code bytes per query x queries / scan time (CPU row: whole search).\n"
              "It counts L2 hits and shared reads (v7 loads each code once per Q queries); use Nsight Compute for "
              "DRAM bytes.");

    const bool exists = std::ifstream(a.csv).good();
    std::ofstream csv(a.csv, std::ios::app);
    if (!exists)
        csv << "label,variant,n,d,m,nq,k,rerank,threads,blocks_x,grid_y,qpb,ilp,blocks_per_sm,waves,lut_ms,scan_ms,"
               "select_ms,transfer_ms,total_ms,qps,code_gbs,recall\n";
    for (const auto& r : rows)
        csv << a.label << ',' << r.variant << ',' << a.n << ',' << a.d << ',' << a.m << ',' << a.nq << ',' << a.k
            << ',' << a.rerank << ',' << r.threads << ',' << r.blocks_x << ',' << r.grid_y << ',' << r.qpb << ',' << r.ilp << ','
            << r.blocks_per_sm << ',' << r.waves << ',' << r.lut_ms << ',' << r.scan_ms << ',' << r.select_ms << ','
            << r.transfer_ms << ',' << r.total_ms << ',' << r.qps << ',' << r.code_gbs << ',' << r.recall << '\n';
    std::printf("appended %zu rows to %s\n", rows.size(), a.csv.c_str());
    return 0;
} catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
}
