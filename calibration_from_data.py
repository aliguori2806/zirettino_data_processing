import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from scipy.stats import linregress
import matplotlib.pyplot as plt

def gaussian(x, a, b, c):
    return a*np.exp(-(x-b)*(x-b)/(2*c*c))

def linear(x, a, b):
    return a*x + b
    
def fit_pedestal_from_data(charges, max_y_secondary_peaks=None, fitfactor=5, range=(0,7000), showplot=False, label=None):
    # Will work as long as the pedestal is the highest peak in the distribution of charges
    # The code considers as peak_half_width the half width of the region where the pedestal is higher than the parameter max_y_secondary_peaks. The fitting region is an interval whose semi-amplitude is peak_half_width*fitfactor.
    # NOTE: max_y_secondary_peaks must be higher than the max y of the highers secondary peak of the charge spectrum
    rl = round(range[0])
    rr = round(range[1])
    nbins = rr - rl + 1 
    hl = rl - 0.5
    hr = rr + 0.5
    y, x = np.histogram(charges, bins=nbins, range=(hl, hr))
    xc = 0.5*(x[:-1] + x[1:])
    xp = xc[np.argmax(y)]
    yp = np.max(y)
    if max_y_secondary_peaks is None:
        max_y_secondary_peaks = yp/2
    xpl = xc[np.where(y > max_y_secondary_peaks)][0]
    xpr = xc[np.where(y > max_y_secondary_peaks)][-1]
    peak_half_width = (xpr - xpl)/2
    fitmask = (xc > xp - fitfactor*peak_half_width) & (xc < xp + fitfactor*peak_half_width)
    par, cov = curve_fit(gaussian, xc[fitmask], y[fitmask], sigma=np.sqrt(y[fitmask]), absolute_sigma=True, p0=[yp, xp, peak_half_width])

    if showplot == True:
        plt.figure(figsize=(8,6))
        plt.step(xc, y, label="Spectrum")
        plt.axvline(xp, label="Spectrum maximum", color="blue")
        plt.axvline(xp - fitfactor*peak_half_width, label="Left edge on fit interval", color="red")
        plt.axvline(xp + fitfactor*peak_half_width, label="Right edge of fit interval", color="red")
        plt.plot(np.linspace(xp - fitfactor*peak_half_width, xp + fitfactor*peak_half_width, 100), gaussian(np.linspace(xp - fitfactor*peak_half_width, xp + fitfactor*peak_half_width, 100), par[0], par[1], par[2]), color="magenta")
        plt.xlim(xp - 2*fitfactor*peak_half_width, xp + 2*fitfactor*peak_half_width)
        plt.legend()
        plt.set_title(label)
        
    return par[1], np.sqrt(cov[1][1]), par[2], np.sqrt(cov[2][2])

def rough_calibration(charges_ps, first_peak, nbins, rl, rr, ped_sigma=3, minheight=15, showplot=False, label=None):

    y, x = np.histogram(charges_ps, bins=nbins, range=(rl, rr))
    xc = 0.5*(x[:-1] + x[1:])

    bin_peak0 = np.digitize(0, x) - 1
    bin_peak1 = np.digitize(first_peak, x) - 1
    distance = round(2*(bin_peak1 - bin_peak0)/3)   
    peaks, prop = find_peaks(y, distance=distance, height=minheight)
    good_peaks = [peak for peak in peaks if xc[peak] > - 3*ped_sigma]
    
    par, cov = curve_fit(linear, np.arange(len(good_peaks)), xc[good_peaks], sigma=0.1*np.abs(xc[good_peaks]), absolute_sigma=True)

    if showplot == True:
        fig, axs = plt.subplots(1, 2, figsize=(16, 6))
        axs = axs.flatten()
        axs[0].hist(charges_ps, bins = nbins, range=(rl, rr), histtype="step")
        axs[0].plot(xc[good_peaks], y[good_peaks], marker="o", color="red")
        axs[0].set_xlabel("Charge (ADC)")
        axs[0].set_ylabel("Entries")
        axs[0].set_yscale("log")
        axs[0].grid()
        axs[0].set_title(label)

        axs[1].errorbar(np.arange(len(good_peaks)), xc[good_peaks], yerr=0.1*xc[good_peaks], marker="o", linestyle="")
        axs[1].plot(np.arange(len(good_peaks)), linear(np.arange(len(good_peaks)), par[0], par[1]), color="red")
        axs[1].annotate(f"gain = {par[0]:.2f}", xy=(0.65, 0.15), xycoords='axes fraction', fontsize=12, ha='center', va='center', bbox=dict(facecolor='lightgray', alpha=0.5))
        axs[1].set_xlabel("Photoelectrons")
        axs[1].set_ylabel("Average Charge (ADC)")
        axs[1].grid()

    return par[0], np.sqrt(cov[0][0]), par[1], np.sqrt(cov[1][1])


def low_gain_cross_calibration(chargesHG_ps, chargesLG_ps, maxHGcut=0.6, showplot=False):

    x = chargesHG_ps
    y = chargesLG_ps
    slope, interc, r_value, p_value, std_err = linregress(x[ (x > 0) & (x < maxHGcut*np.max(x))], y[(x > 0) & (x < maxHGcut*np.max(x))])

    if showplot == True:
        fig, axs = plt.subplots(1, 1, figsize=(8, 6))
        axs.scatter(x, y, s=0.5)
        axs.plot(x[ (x > 0) & (x < maxHGcut*np.max(x))], slope*x[ (x > 0) & (x < maxHGcut*np.max(x))] + interc, color="red")
        axs.set_xlabel("HG (ADC)")
        axs.set_ylabel("LG (ADC)")
        axs.annotate(f"1/LGf = {1/slope:.3f}", xy=(0.65, 0.15), xycoords='axes fraction',
                fontsize=12, ha='center', va='center',
                bbox=dict(facecolor='lightgray', alpha=0.5))
        axs.grid()
    
    return slope
    
    
    