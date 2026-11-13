from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd

EXPERIMENT_FOLDER = Path(__file__).resolve().parent
SAMPLES_FOLDER = EXPERIMENT_FOLDER.parent / "exp3" / "samples"
FIGURE_PATH = EXPERIMENT_FOLDER / "figures" / "fig7_trajectory_statistics.png"

FIT_FIGURE_PATH = EXPERIMENT_FOLDER / "figures" / "fig8_distribution_fits.png"      
PARAMETER_FIGURE_PATH = EXPERIMENT_FOLDER / "figures" / "fig9_fitted_parameters.png"
RESULTS_PATH = EXPERIMENT_FOLDER / "fitted_parameters.csv"                          

MINIMUM_DAYS = 60
CONFIGS = ["A", "B", "C"]
SOURCES = ["Real"] + CONFIGS
SEEDS = ["s1", "s2", "s3"]
STATISTICS = ["daily_returns", "volatility", "max_drawdown"]
PLOT_COLOURS = {"Real": "black", "A": "tab:blue", "B": "tab:orange", "C": "tab:green"}
PLOT_TITLES = {"daily_returns": "daily log returns (log scale)",
               "volatility": "volatility per path",
               "max_drawdown": "max drawdown per path"}


N_BOOTSTRAP = 200                                                            
MAX_FIT_SAMPLES = 100_000                                                    
RANDOM_GENERATOR = np.random.default_rng(0)                                  
CANDIDATES = {                                                               
    "daily_returns": {"norm": stats.norm, "laplace": stats.laplace, "t": stats.t,         
                      "gennorm": stats.gennorm, "johnsonsu": stats.johnsonsu},            
    "volatility": {"lognorm": stats.lognorm, "gamma": stats.gamma, "invgamma": stats.invgamma,   
                   "weibull_min": stats.weibull_min, "fisk": stats.fisk},                        
    "max_drawdown": {"beta": stats.beta, "lognorm": stats.lognorm, "gamma": stats.gamma,         
                     "weibull_min": stats.weibull_min},                                          
}                                                                   
FORCE_FAMILY = {}                                                   



def statistics_of_paths(paths):
    daily_returns, volatility, max_drawdown = [], [], []
    for prices in paths:
        log_returns = np.diff(np.log(prices))
        daily_returns.append(log_returns)
        volatility.append(log_returns.std())
        max_drawdown.append(np.max(1 - prices / np.maximum.accumulate(prices)))
    return {"daily_returns": np.concatenate(daily_returns),
            "volatility": np.array(volatility),
            "max_drawdown": np.array(max_drawdown)}


reference_samples = np.load(SAMPLES_FOLDER / "Exp2_A_s1.npz")
real_lengths = reference_samples["original_price_len"]
real_prices = reference_samples["original_price"]
scenarios_to_use = np.where(real_lengths >= MINIMUM_DAYS)[0]
# real_statistics = statistics_of_paths([real_prices[scenario, :real_lengths[scenario]] for scenario in scenarios_to_use])

real_path_list = [real_prices[scenario, :real_lengths[scenario]] for scenario in scenarios_to_use]   
real_statistics = statistics_of_paths(real_path_list)                                                



statistics_per_run, distances_per_run = {}, {}
for config in CONFIGS:
    for seed in SEEDS:
        samples = np.load(SAMPLES_FOLDER / f"Exp2_{config}_{seed}.npz")
        synthetic_paths = samples["synthetic_paths"]
        scenario_of_path = samples["base_sample_id"]
        path_is_valid = samples["is_valid"]
        valid_paths = []
        for path_index, scenario in enumerate(scenario_of_path):
            prices = synthetic_paths[path_index, :real_lengths[scenario]]
            if path_is_valid[path_index] and real_lengths[scenario] >= MINIMUM_DAYS and np.all(prices > 0):
                valid_paths.append(prices)
        run = (config, seed)
        statistics_per_run[run] = statistics_of_paths(valid_paths)
        distances_per_run[run] = {name: stats.wasserstein_distance(statistics_per_run[run][name], real_statistics[name])
                                  for name in STATISTICS}
        print(f"{config}_{seed} ({len(valid_paths)} paths): "
              + "  ".join(f"{name} {value:.4f}" for name, value in distances_per_run[run].items()))

print("\nDistance to real, mean +- std over seeds (0 = identical to real)")
for config in CONFIGS:
    summary_parts = []
    for name in STATISTICS:
        values = [distances_per_run[(config, seed)][name] for seed in SEEDS]
        summary_parts.append(f"{name} {np.mean(values):.4f}+-{np.std(values, ddof=1):.4f}")
    print(config, "  ".join(summary_parts))

print("\nReturn std per seed (real {:.4f})".format(real_statistics["daily_returns"].std()))
for config in CONFIGS:
    print(config, "  ".join(f"{seed} {statistics_per_run[(config, seed)]['daily_returns'].std():.4f}" for seed in SEEDS))

pooled_statistics = {"Real": real_statistics}
for config in CONFIGS:
    pooled_statistics[config] = {name: np.concatenate([statistics_per_run[(config, seed)][name] for seed in SEEDS])
                                 for name in STATISTICS}

print("\nTypical values")
for label, statistics in pooled_statistics.items():
    print(f"{label:5} return std {statistics['daily_returns'].std():.4f}  "
          f"return excess kurtosis {stats.kurtosis(statistics['daily_returns']):.2f}  "
          f"mean volatility {statistics['volatility'].mean():.4f}  "
          f"median max drawdown {np.median(statistics['max_drawdown']):.3f}")

figure, axes = plt.subplots(1, 3, figsize=(16, 5))
for axis, name in zip(axes, STATISTICS):
    all_values = np.concatenate([statistics[name] for statistics in pooled_statistics.values()])
    histogram_bins = np.linspace(*np.percentile(all_values, [0.5, 99.5]), 60)
    for label, statistics in pooled_statistics.items():
        axis.hist(statistics[name], bins=histogram_bins, density=True, histtype="step",
                  linewidth=1.5, color=PLOT_COLOURS[label], label=label)
    axis.set_title(PLOT_TITLES[name])
axes[0].set_yscale("log")
axes[0].legend()
figure.suptitle("Experiment 4: real vs generated paths (3 seeds)")
figure.tight_layout()
figure.savefig(FIGURE_PATH, dpi=150)





def fit_distribution(distribution, statistic, data):                         
    fit_kwargs = {}                                                          
    if statistic != "daily_returns":                                         
        fit_kwargs["floc"] = 0                                               
    if distribution is stats.beta:                                           
        fit_kwargs["fscale"] = 1                                             
    return distribution.fit(data, **fit_kwargs)                              


def parameter_names(distribution, statistic):                                
    shapes = distribution.shapes.split(", ") if distribution.shapes else []  
    all_names = shapes + ["loc", "scale"]                                    
    free_names = list(all_names)                                             
    if statistic != "daily_returns":                                         
        free_names.remove("loc")                                             
    if distribution is stats.beta:                                           
        free_names.remove("scale")                                           
    return all_names, free_names                                             


def free_parameter_values(distribution, statistic, params):           
    all_names, free_names = parameter_names(distribution, statistic)  
    by_name = dict(zip(all_names, params))                            
    return {name: by_name[name] for name in free_names}               


fit_rows, pooled_fits = [], {}                                               
for statistic in STATISTICS:                                                 
    for source in SOURCES:                                                   
        data = pooled_statistics[source][statistic]                          
        if len(data) > MAX_FIT_SAMPLES:                                      
            data = RANDOM_GENERATOR.choice(data, MAX_FIT_SAMPLES, replace=False)
        for family, distribution in CANDIDATES[statistic].items():           
            try:                                                             
                params = fit_distribution(distribution, statistic, data)     
                log_likelihood = distribution.logpdf(data, *params).sum()    
                n_free = len(parameter_names(distribution, statistic)[1])    
                ks_statistic = stats.kstest(data, distribution.cdf, args=params).statistic   
                fit_rows.append({"statistic": statistic, "source": source, "family": family, 
                                 "aic": 2 * n_free - 2 * log_likelihood, "ks": ks_statistic})
                pooled_fits[(statistic, source, family)] = params            
            except Exception as error:                                       
                print(f"fit failed: {statistic} {source} {family}: {error}") 

fit_table = pd.DataFrame(fit_rows)                                           
fit_table["delta_aic"] = fit_table["aic"] - fit_table.groupby(["statistic", "source"])["aic"].transform("min")

chosen_family = {}                                                           
for statistic in STATISTICS:                                                 
    subset = fit_table[fit_table["statistic"] == statistic]                  
    print(f"\n=== {statistic}: delta AIC (0 = best for that source; lower is better) ===")
    print(subset.pivot(index="family", columns="source", values="delta_aic")[SOURCES].round(1).to_string())
    print(f"--- {statistic}: KS statistic (distance between fit and data CDF) ---")           
    print(subset.pivot(index="family", columns="source", values="ks")[SOURCES].round(4).to_string())
    summed_delta = subset.groupby("family")["delta_aic"].sum()               
    chosen_family[statistic] = FORCE_FAMILY.get(statistic, summed_delta.idxmin())
    print(f"--> chosen family for {statistic}: {chosen_family[statistic]}")  



def fit_chosen(statistic, data):                                             
    distribution = CANDIDATES[statistic][chosen_family[statistic]]           
    return free_parameter_values(distribution, statistic, fit_distribution(distribution, statistic, data))


real_point = {statistic: fit_chosen(statistic, real_statistics[statistic]) for statistic in STATISTICS}

print(f"\nBootstrapping real paths ({N_BOOTSTRAP} resamples) ...")            
real_bootstrap = {statistic: {name: [] for name in real_point[statistic]} for statistic in STATISTICS}   
for _ in range(N_BOOTSTRAP):                                                 
    resampled = RANDOM_GENERATOR.integers(0, len(real_path_list), len(real_path_list))  
    resampled_statistics = statistics_of_paths([real_path_list[index] for index in resampled])
    for statistic in STATISTICS:                                             
        try:                                                                 
            for name, value in fit_chosen(statistic, resampled_statistics[statistic]).items():
                real_bootstrap[statistic][name].append(value)                
        except Exception:                                                    
            pass                                                             

per_run_parameters = {statistic: {(config, seed): fit_chosen(statistic, statistics_per_run[(config, seed)][statistic])   
                                  for config in CONFIGS for seed in SEEDS}   
                      for statistic in STATISTICS}                           


summary_rows = []                                                            
for statistic in STATISTICS:                                                 
    for name, real_value in real_point[statistic].items():                   
        boot = np.array(real_bootstrap[statistic][name])                     
        real_standard_error = boot.std(ddof=1)                               
        row = {"statistic": statistic, "family": chosen_family[statistic], "parameter": name,
               "real": real_value, "real_boot_se": real_standard_error,      
               "real_ci_low": np.percentile(boot, 2.5), "real_ci_high": np.percentile(boot, 97.5)}
        for config in CONFIGS:                                               
            values = [per_run_parameters[statistic][(config, seed)][name] for seed in SEEDS]
            row[f"{config}_mean"] = np.mean(values)                          
            row[f"{config}_std"] = np.std(values, ddof=1)                    
            row[f"{config}_z_vs_real"] = (np.mean(values) - real_value) / real_standard_error   
        summary_rows.append(row)                                            

summary_table = pd.DataFrame(summary_rows)                                  
summary_table.to_csv(RESULTS_PATH, index=False)                             
print("\n=== Fitted parameters (z = (config - real) / real SE) ===")        
print(summary_table.round(4).to_string())                                   
print(f"\nSaved {RESULTS_PATH}")                                            


# fig8: histograms with the fitted pdf on top (dashed)
figure, axes = plt.subplots(1, 3, figsize=(16, 5))                           
for axis, statistic in zip(axes, STATISTICS):                                
    family = chosen_family[statistic]                                        
    distribution = CANDIDATES[statistic][family]                             
    all_values = np.concatenate([pooled_statistics[source][statistic] for source in SOURCES])
    histogram_bins = np.linspace(*np.percentile(all_values, [0.5, 99.5]), 60)   
    grid = np.linspace(histogram_bins[0], histogram_bins[-1], 400)           
    for source in SOURCES:                                                   
        axis.hist(pooled_statistics[source][statistic], bins=histogram_bins, density=True, histtype="step",   
                  linewidth=1.0, color=PLOT_COLOURS[source], alpha=0.6, label=source)                         
        axis.plot(grid, distribution.pdf(grid, *pooled_fits[(statistic, source, family)]),                    
                  color=PLOT_COLOURS[source], linestyle="--", linewidth=1.6)                                  
    axis.set_title(f"{PLOT_TITLES[statistic]}\nfit: {family} (dashed)")      
axes[0].set_yscale("log")                                                    
axes[0].set_ylim(bottom=0.05)                                                
axes[0].legend()                                                             
figure.suptitle("Experiment 4b: fitted distributions, real vs generated (pooled over 3 seeds)")   
figure.tight_layout()                                                        
figure.savefig(FIT_FIGURE_PATH, dpi=150)                                     
plt.close(figure)                                                            

# fig9: fitted parameters per config (dots = seeds, bar = mean +- std) against the real 95% bootstrap interval
max_parameters = max(len(real_point[statistic]) for statistic in STATISTICS) 
figure, axes = plt.subplots(len(STATISTICS), max_parameters, figsize=(4.2 * max_parameters, 3.6 * len(STATISTICS)),   
                            squeeze=False)                                   
for row_index, statistic in enumerate(STATISTICS):                           
    names = list(real_point[statistic])                                      
    for column_index in range(max_parameters):                               
        axis = axes[row_index, column_index]                                 
        if column_index >= len(names):                                       
            axis.axis("off")                                                 
            continue                                                         
        name = names[column_index]                                           
        summary = summary_table[(summary_table["statistic"] == statistic) & (summary_table["parameter"] == name)].iloc[0]
        axis.axhspan(summary["real_ci_low"], summary["real_ci_high"], color="black", alpha=0.12,
                     label="real 95% bootstrap CI")                          
        axis.axhline(summary["real"], color="black", linewidth=1.2, label="real")  
        for config_index, config in enumerate(CONFIGS):                      
            values = [per_run_parameters[statistic][(config, seed)][name] for seed in SEEDS]  
            axis.scatter([config_index] * len(values), values, color=PLOT_COLOURS[config], s=28, alpha=0.7)  
            axis.errorbar(config_index, np.mean(values), yerr=np.std(values, ddof=1), color=PLOT_COLOURS[config],  
                          marker="_", markersize=18, capsize=4)              
        axis.set_xticks(range(len(CONFIGS)))                                 
        axis.set_xticklabels(CONFIGS)                                        
        axis.set_title(f"{statistic}: {chosen_family[statistic]} {name}")    
axes[0, 0].legend(fontsize=8)                                                
figure.suptitle("Experiment 4b: fitted parameters per config (dots = seeds, bar = mean +- std) vs real")   
figure.tight_layout()                                                        
figure.savefig(PARAMETER_FIGURE_PATH, dpi=150)                               
plt.close(figure)                                                            
print(f"Saved {FIT_FIGURE_PATH}\nSaved {PARAMETER_FIGURE_PATH}")             