import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from arch import arch_model
from arch.univariate import ARX
from scipy.stats import ks_2samp
from statsmodels.tsa.stattools import acf

from concurrent.futures import ThreadPoolExecutor


###############################################################################
#                 OPTIONAL HELPER: REMOVE OR CLAMP OUTLIER RETURNS           #
###############################################################################
def winsorize_log_returns(df, z_thresh=4.0):
    """
    Clamp extreme log-returns in the DataFrame's close prices.
    This helps reduce the impact of outliers on GARCH fitting.
    
    Parameters
    ----------
    df : pd.DataFrame
        Must have a 'close' column.
    z_thresh : float
        Z-score threshold for clamping. Default=4.0 => ~4 std devs.
        
    Returns
    -------
    df_winsor : pd.DataFrame
        A copy of df with extreme log-returns clamped.
    """
    df_winsor = df.copy()
    # Compute log-returns
    logp = np.log(df_winsor['close'])
    log_ret = logp.diff()

    # Identify outliers via z-scores
    mean_lr = log_ret.mean()
    std_lr  = log_ret.std()
    upper_bound = mean_lr + z_thresh * std_lr
    lower_bound = mean_lr - z_thresh * std_lr

    # Clamp the log-returns
    log_ret_clamped = log_ret.clip(lower_bound, upper_bound)

    # Reconstruct prices from clamped log-returns
    # Start from the same initial log-price
    new_logp = [logp.iloc[0]]
    for i in range(1, len(logp)):
        new_logp.append(new_logp[-1] + log_ret_clamped.iloc[i])

    # Exponentiate back
    df_winsor['close'] = np.exp(new_logp)

    return df_winsor

###############################################################################
#                          1. GARCH-BASED SYNTHESIZER                         #
###############################################################################
class GarchSynthesizer:
    """
    Synthesizes price data via a GARCH model on log-returns.
    Reconstructs OHLC using percentage ratios computed from the original data.
    The open price for each bar is set equal to the previous bar's synthetic close.
    """
    def __init__(
            self,
            original_df,
            dist='StudentsT',
            p=1,
            q=1,
            o=1,
            power=2.0,
            mean_type='Constant',
            use_volume=False,
            lags=3,
            rescale=False,
            clamp_extremes=True,
            max_log_return=1.0,
            min_log_return=-1.0,
            burn=500
    ):
        self.original_df = original_df.copy()
        self.dist = dist
        self.p = p
        self.q = q
        self.o = o
        self.power = power
        self.mean_type=mean_type
        self.use_volume=use_volume
        self.lags=lags
        self.rescale = rescale
        self.clamp_extremes = clamp_extremes
        self.max_log_return = max_log_return
        self.min_log_return = min_log_return
        self.burn = burn

        # Verify required columns are present
        required_cols = ['close', 'high', 'low', 'open', 'volume']
        for col in required_cols:
            if col not in self.original_df.columns:
                raise ValueError(f"Column '{col}' not found in original_df.")
        
        
        # Save percentage ratios for OHLC reconstruction:
        # high_ratio = high / open, low_ratio = low / high.
        self.original_df['high_ratio'] = self.original_df['high'] / self.original_df['open']
        self.original_df['low_ratio']  = self.original_df['low']  / self.original_df['high']

        # Use log-returns to reduce numerical issues
        log_prices = np.log(self.original_df['close'])
        self.returns = log_prices.diff().dropna()

        if self.use_volume:
            vol_p = self.original_df['volume'].pct_change().dropna()
            # Align vol_p with self.returns index
            self.vol_p = vol_p.loc[self.returns.index]
            # Fit a simpler GARCH(1,1) with normal distribution
            self.model = arch_model(
                self.returns,
                p=self.p,
                q=self.q,
                o=self.o,
                power=self.power,
                vol='Garch',
                dist=self.dist,
                rescale=self.rescale,
                mean='ARX',
                x=self.vol_p,
                lags=self.lags,
            )
        else:
            # Fit a simpler GARCH(1,1) with normal distribution
            self.model = arch_model(
                self.returns,
                p=self.p,
                q=self.q,
                o=self.o,
                power=self.power,
                vol='Garch',
                dist=self.dist,
                rescale=self.rescale,
                mean= self.mean_type  # or 'Zero', 'AR'
            )

        self.result = self.model.fit(update_freq=5)
        print(self.result.summary())

    def synthesize_data(self, M=1, start_price=None):
        """
        Generate M synthetic DataFrames using the fitted GARCH model concurrently.
        """
        original_close = self.original_df['close'].values
        n = len(self.returns)
    
        def simulate_once():
            # Perform a single simulation
            if self.use_volume:
                # Use the complete parameter vector (do not drop the exogenous parameter)
                params_sim = self.result.params
                exog_orig = self.vol_p.to_frame()
                # Create a DataFrame with burn rows using the first row of exog_orig
                pad_df = pd.DataFrame(
                    np.tile(exog_orig.iloc[0].values, (self.burn, 1)),
                    columns=exog_orig.columns
                )
                # Concatenate the padding with the original exogenous data
                exog_extended = pd.concat([pad_df, exog_orig], ignore_index=True)
                sim_res = self.model.simulate(
                    params=params_sim,
                    nobs=n,
                    burn=self.burn,
                    x=exog_extended  # Now has (nobs + burn) rows
                )
            else:
                sim_res = self.model.simulate(
                    params=self.result.params,
                    nobs=n,
                    burn=self.burn
                )
            
            simulated_log_returns = sim_res['data'].values
    
            # Build synthetic close prices
            synthetic_close = []
            if start_price is None:
                synthetic_close.append(original_close[0])
            else:
                synthetic_close.append(start_price)
            for r in simulated_log_returns:
                if self.clamp_extremes:
                    r = np.clip(r, self.min_log_return, self.max_log_return)
                next_price = synthetic_close[-1] * np.exp(r)
                synthetic_close.append(next_price)
            # synthetic_close now has T values (n+1), same as the original dataset
    
            # Compute synthetic open: for day 0, we set open=close; then use previous close for subsequent days.
            synthetic_open = [synthetic_close[0]] + synthetic_close[:-1]
    
            # Use saved percentage ratios to compute synthetic high and low
            high_ratios = self.original_df['high_ratio'].values
            low_ratios  = self.original_df['low_ratio'].values
    
            synthetic_high = [o * hr for o, hr in zip(synthetic_open, high_ratios)]
            synthetic_low  = [h * lr for h, lr in zip(synthetic_high, low_ratios)]
    
            # Build the synthetic DataFrame with the same index as the original
            synth_df = pd.DataFrame({
                'open': synthetic_open,
                'high': synthetic_high,
                'low': synthetic_low,
                'close': synthetic_close,
                'volume': self.original_df['volume']
            }, index=self.original_df.index)
            return synth_df
    
        # Use a ThreadPoolExecutor to run M simulations concurrently.
        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = [executor.submit(simulate_once) for _ in range(M)]
            # Wait for all threads to complete and collect their results
            new_dataframes = [future.result() for future in futures]
    
        return new_dataframes

###############################################################################
#             REGIME-ADAPTIVE GARCH SYNTHESIZER (OPTIONAL CLASS)              #
###############################################################################
class RegimeAdaptiveGarchSynthesizer:
    """
    Splits the original data into contiguous chunks (regimes),
    fits a separate GARCH model to each chunk, and chains the synthetic series.
    """
    def __init__(self, original_df, n_chunks, **garch_kwargs):
        # Ensure original_df is sorted by its index
        original_df = original_df.sort_index()
        self.chunk_dfs = np.array_split(original_df, n_chunks)
        self.models = [GarchSynthesizer(chunk, **garch_kwargs) for chunk in self.chunk_dfs]
    
    def synthesize_data(self, M=1):
        """
        Generate M synthetic DataFrames by sequentially simulating each regime.
        The ending synthetic close of one regime is used as the starting price for the next.
        """
        all_synths = []
        for _ in range(M):
            synthetic_chunks = []
            current_start = None
            for model in self.models:
                # Chain regimes using current_start as the starting price for each chunk.
                synth_chunk = model.synthesize_data(M=1, start_price=current_start)[0]
                current_start = synth_chunk['close'].iloc[-1]
                synthetic_chunks.append(synth_chunk)
            # Concatenate all chunks to form the full synthetic dataset.
            full_synth = pd.concat(synthetic_chunks)
            all_synths.append(full_synth)
        return all_synths

###############################################################################
#                       2. DATA QUALITY TESTER                                #
###############################################################################
class DataQualityTester:
    """
    Compares original and synthetic data using statistical tests.
    """
    def __init__(self, original_df, synthetic_df):
        self.original_df   = original_df
        self.synthetic_df  = synthetic_df
        self.original_returns  = self.compute_returns(original_df['close'])
        self.synthetic_returns = self.compute_returns(synthetic_df['close'])

    def compute_returns(self, series):
        """Compute simple percentage returns from a price series."""
        return series.pct_change().dropna()

    def distribution_test_ks(self):
        """Perform a two-sample Kolmogorov-Smirnov test on returns distributions."""
        stat, pvalue = ks_2samp(self.original_returns, self.synthetic_returns)
        return stat, pvalue

    def mean_std_test(self):
        """Compare mean and standard deviation of returns."""
        orig_mean = self.original_returns.mean()
        synth_mean = self.synthetic_returns.mean()
        orig_std = self.original_returns.std()
        synth_std = self.synthetic_returns.std()
        return {
            'original_mean':  orig_mean,
            'synthetic_mean': synth_mean,
            'original_std':   orig_std,
            'synthetic_std':  synth_std
        }

    def autocorrelation_test(self, max_lag=10):
        """Compare autocorrelations (ACF) of original and synthetic returns."""
        orig_acf_vals  = acf(self.original_returns, nlags=max_lag, fft=False)
        synth_acf_vals = acf(self.synthetic_returns, nlags=max_lag, fft=False)
        return pd.DataFrame({
            'lag': np.arange(max_lag+1),
            'original_acf': orig_acf_vals,
            'synthetic_acf': synth_acf_vals
        })

    def run_all_tests(self, verbose=False):
        """Run multiple tests and return aggregated results."""
        all_results = {}

        # KS Test
        ks_stat, ks_pvalue = self.distribution_test_ks()
        all_results['ks_stat']   = ks_stat
        all_results['ks_pvalue'] = ks_pvalue
        if verbose:
            print(f"[KS Test] stat={ks_stat:.4f}, p-value={ks_pvalue:.4f}")

        # Mean/Std comparison
        ms_results = self.mean_std_test()
        all_results.update(ms_results)
        if verbose:
            print(f"[Mean/Std] Original mean={ms_results['original_mean']:.5f}, std={ms_results['original_std']:.5f}")
            print(f"           Synthetic mean={ms_results['synthetic_mean']:.5f}, std={ms_results['synthetic_std']:.5f}")

        # Autocorrelation test
        acf_comp = self.autocorrelation_test()
        all_results['acf_comparison'] = acf_comp

        return all_results

###############################################################################
#                           3. PLOTTING FUNCTIONS                             #
###############################################################################
def plot_synthetic_data(original_df, synthetic_dfs, title="GARCH + Log-Returns Synthesis", num_to_plot=3):
    plt.figure(figsize=(12,6))
    plt.plot(original_df.index, original_df['close'], label="Original Close", color='black', lw=2)
    for i, synth in enumerate(synthetic_dfs[:num_to_plot]):
        plt.plot(synth.index, synth['close'], label=f"Synth {i+1}", alpha=0.7)
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_test_summaries(results_list):
    ks_stats   = [r['ks_stat']   for r in results_list]
    ks_pvalues = [r['ks_pvalue'] for r in results_list]
    synth_means= [r['synthetic_mean'] for r in results_list]
    synth_stds = [r['synthetic_std']  for r in results_list]

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.ravel()

    # KS Statistic Histogram
    axes[0].hist(ks_stats, bins=20, alpha=0.7, color='blue')
    axes[0].axvline(np.mean(ks_stats), color='black', linestyle='--')
    axes[0].set_title("Distribution of KS Statistics")
    axes[0].set_xlabel("KS Statistic")

    # KS p-value Histogram
    axes[1].hist(ks_pvalues, bins=20, alpha=0.7, color='green')
    axes[1].axvline(np.mean(ks_pvalues), color='black', linestyle='--')
    axes[1].set_title("Distribution of KS p-values")
    axes[1].set_xlabel("KS p-value")

    # Synthetic Mean Histogram
    axes[2].hist(synth_means, bins=20, alpha=0.7, color='orange')
    axes[2].axvline(np.mean(synth_means), color='black', linestyle='--')
    axes[2].set_title("Distribution of Synthetic Means")
    axes[2].set_xlabel("Mean of Returns")

    # Synthetic Std Dev Histogram
    axes[3].hist(synth_stds, bins=20, alpha=0.7, color='red')
    axes[3].axvline(np.mean(synth_stds), color='black', linestyle='--')
    axes[3].set_title("Distribution of Synthetic Std Dev")
    axes[3].set_xlabel("Std Dev of Returns")

    plt.tight_layout()
    plt.show()

    print("=== Aggregated Test Results ===")
    print(f"Avg KS Stat:    {np.mean(ks_stats):.4f}")
    print(f"Avg KS p-value: {np.mean(ks_pvalues):.4f}")
    print(f"Avg Synthetic Mean (returns): {np.mean(synth_means):.6f}")
    print(f"Avg Synthetic Std  (returns): {np.mean(synth_stds):.6f}")

###############################################################################
#                           4. MAIN EXECUTION FLOW                            #
###############################################################################
if __name__ == '__main__':
    excel_file = 'crypto_data_1D_2017-01-01_to_2025-01-18_20250118_091237.xlsx'
    sheet_name = 'BTC'
    try:
        original_data = pd.read_excel(excel_file, sheet_name=sheet_name)
    except FileNotFoundError:
        print(f"Error: Could not find {excel_file}")
        exit()
    except KeyError:
        print(f"Error: Sheet {sheet_name} not found.")
        exit()

    # Ensure lowercase column names and proper datetime indexing
    original_data.columns = original_data.columns.str.lower()
    if 'datetime' not in original_data.columns:
        print("Error: 'datetime' column not found.")
        exit()

    original_data['datetime'] = pd.to_datetime(original_data['datetime'])
    original_data.set_index('datetime', inplace=True)
    
    # OPTIONAL: Remove or clamp outlier log-returns before modeling
    original_data = winsorize_log_returns(original_data, z_thresh=4.0)
    
    # Set this flag to choose between a single-regime or regime-adaptive synthesizer.
    use_regime_adaptive = False  # Set to False to use the standard synthesizer

    if use_regime_adaptive:
        # For example, split the data into 4 regimes
        regime_model = RegimeAdaptiveGarchSynthesizer(
            original_df=original_data,
            n_chunks=4,
            dist='skewt',  # or 'normal' or skewt, StudentsT
            p=3, q=3, o=1,
            power=1.0, mean_type='Constant',  # or 'Zero', 'AR'
            use_volume=True, #if True discard mean_type and use ARX
            lags=3,
            rescale=False,
            clamp_extremes=True,
            max_log_return=1.0,
            min_log_return=-1.0,
            burn=200
        )
        synthetic_dfs = regime_model.synthesize_data(M=300)
    else:
        garch_synth = GarchSynthesizer(
            original_df=original_data,
            dist='skewt',  # or 'normal' or skewt, StudentsT
            p=3, q=3, o=1,
            power=1.0, mean_type='Constant',  # or 'Zero', 'AR'
            use_volume=True, #if True discard mean_type and use ARX
            lags=3,
            rescale=False,
            clamp_extremes=True,
            max_log_return=1.0,
            min_log_return=-1.0,
            burn=200
        )
        synthetic_dfs = garch_synth.synthesize_data(M=300)

    # Use the full original data for plotting and tests.
    aligned_original = original_data.copy()

    # Plot a few synthetic paths
    plot_synthetic_data(aligned_original, synthetic_dfs, title="GARCH + Log-Returns Synthesis", num_to_plot=10)

    # Evaluate quality of each synthetic dataset
    results_list = []
    for synth_df in synthetic_dfs:
        tester = DataQualityTester(aligned_original, synth_df)
        results = tester.run_all_tests(verbose=False)
        results_list.append(results)

    # Summarize the test results
    plot_test_summaries(results_list)
