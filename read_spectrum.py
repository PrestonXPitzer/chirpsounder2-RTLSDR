from rtlsdr import RtlSdr
sdr = RtlSdr()
#Basic device configuration
#sdr.device_index = 0 # May be needed if multiple devices are connected
#(This may be necessary if we need additional bandwidth!)
sdr.sample_rate = 2.4e6  # or lower if needed
sdr.center_freq = 14.074e6 # This is the 20m FT8 frequency, so there should be some action here
sdr.gain = 'auto'
#These settings are to optimize HF reception
sdr.set_agc_mode(True) #Enable Automatic Gain Control
sdr.set_direct_sampling(2) #Enable direct sampling mode (Q channel)
samples = sdr.read_samples(256*1024)

from scipy.signal import spectrogram
import matplotlib.pyplot as plt
import numpy as np

f, t, Sxx = spectrogram(samples, fs=sdr.sample_rate, nperseg=2048, noverlap=1024)
plt.pcolormesh(f, t, 10*np.log10(np.abs(Sxx)).T, shading='gouraud')  # Transpose Sxx and swap axes
plt.xlabel('Frequency [Hz]')
plt.ylabel('Time [sec]')
plt.title('Spectrogram')
plt.gca().invert_yaxis()  # Invert the time axis
plt.savefig("spectrogram.png")

from skimage.transform import hough_line, hough_line_peaks

# Convert spectrogram to 2D image
spectrogram_db = 10 * np.log10(np.abs(Sxx))
spectrogram_img = (spectrogram_db - spectrogram_db.min()) / (spectrogram_db.max() - spectrogram_db.min())

# Apply Hough transform
h, theta, d = hough_line(spectrogram_img)
accums, angles, dists = hough_line_peaks(h, theta, d)

# Lines detected (i.e., chirps)
for angle, dist in zip(angles, dists):
    print(f"Detected chirp with angle={angle:.2f}, dist={dist:.2f}")

# Assume you detect a chirp with sweep from f_start to f_end
# For each frequency bin in the spectrogram (f), find the time index (t) with peak power

ionogram_frequencies = []
ionogram_delays = []

for i, freq_bin in enumerate(f):
    power_slice = Sxx[i, :]
    time_index = np.argmax(power_slice)
    ionogram_frequencies.append(freq_bin)
    ionogram_delays.append(t[time_index])

# Plot Ionogram
plt.plot(ionogram_frequencies, ionogram_delays)
plt.xlabel("Frequency [Hz]")
plt.ylabel("Delay [s]")
plt.title("Ionogram")
plt.grid()
plt.savefig("ionogram.png")
