import numpy as np
from nengo import *
from nengo_spa import *
from typing import *
import matplotlib.pyplot as plt


# I needed a Probe that also stores the vocabulary of the thing it's probing. Ideally nengo should provide one of these by default but whatever..
# If I only use these probes, then I don't have to store the vocabularies separately, and I can plot pointer similarities without passing around a
# bunch of parameters.
class MyProbe(Probe):
	# target:  The thing to probe.
	# label:   How the plot will be labelled.
	# synapse: Does some averaging over the given period of time using a lowpass filter.
	# vocab:   When probing a semantic pointer you should provide here a vocabulary that will be used to interpret the probed values during plotting.
	# mapping: When probing something with multiple dimensions you should give a list of labels for each dimension.
	# label_spikes_in_plots: Whether to label each spike in the plot using the vocab or mapping parameters.
	def __init__(self, target: Union[Node, Ensemble], label: str, *, synapse: float = 0, vocab: Vocabulary = None, mapping: Iterable[str] = None, label_spikes_in_plots: bool = True):
		super().__init__(target, label=label, synapse=synapse if synapse > 0 else None)
		self.vocab = vocab
		self.label_spikes_in_plots = label_spikes_in_plots
		if mapping is not None:
			self.mapping = list(mapping)
		elif mapping is None and vocab is not None:
			self.mapping = list(vocab.keys())
		else:
			self.mapping = None


def plot_simulation_output(title: str, sim: Simulator):
	probes = sim.model.probes
	num_probes = len(probes)

	FONT_SIZE  = 14 # For title, x-axis, y-axis.
	LABEL_SIZE = 12 # For spike labels.

	plt.rc('font', size=FONT_SIZE)
	fig, ax = plt.subplots(num_probes, 1, sharex=True, figsize=(8, 6))

	last_legend = None
	for i, probe in enumerate(probes):
		label = probe.label
		if hasattr(probe, 'vocab'): # MyProbe will have this, normal Probe won't.
			vocab = probe.vocab
		else:
			vocab = None
		ax[i].set_ylabel(label)
		
		# If we have a vocabulary, then we need to actually plot the *similarity* to the semantic pointers - otherwise we get a total mess.
		t = sim.trange()
		y = sim.data[probe]
		if vocab is not None:
			y = similarity(y, vocab)
		ax[i].plot(t, y)

		# Make a legend, but only if the previous plot didn't have the same legend - otherwise everything gets too cluttered.
		legend = probe.mapping
		if legend is not None and legend != last_legend:
			#ax[i].legend(legend, loc='center left', bbox_to_anchor=(1.0, 0.5))
			last_legend = legend

		# Label every spike in this particular plot, because having to constantly look at the legend to determine what each spike is
		# is really annoying. This is probably a bit over-engineered.
		if legend is not None and probe.label_spikes_in_plots and len(np.shape(y)) > 1: # Utility is too noisy to label
			y = np.asarray(y)
			max_y       = np.max(y)
			max_indices = np.argmax(y, axis=1)
			max_values  = np.amax(y, axis=1)
			
			# Find out where each spike starts, how long it lasts, and which values are actually spiking (indices).
			# All of the numpy makes this a bit hard to understand, but the concept of what is being done is pretty simple.
			# I probably had to do it with numpy for the performance, even though simple loops would be a lot more readable.
			# If we run the simulation for like 20 seconds, and we also have 20 probes with 5 values each, that's like 2 million values
			# so this needs to at least be reasonably fast.
			n = len(max_indices)
			spike_start_markers     = np.empty(len(max_indices), dtype=bool)
			spike_start_markers[0]  = True
			spike_start_markers[1:] = max_indices[:-1] != max_indices[+1:]
			
			spike_starts  = np.nonzero(spike_start_markers)[0]
			spike_indices = max_indices[spike_starts]
			spike_lengths = np.diff(np.append(spike_starts, n))

			simulation_start_time = t[0]
			simulation_total_time = t[-1] - t[0]
			time_increment = t[1] - t[0]

			for spike in range(len(spike_starts)):
				start    = spike_starts[spike]
				index    = spike_indices[spike]
				length   = spike_lengths[spike]
				label    = legend[index]
				
				# Don't label very short spikes
				duration = length * time_increment
				if duration > simulation_total_time / 100:

					spike_values = max_values[start : start + length]

					# Don't label a spike unless it's taller than some threshold.
					mean_height = np.mean(spike_values)
					if mean_height > 0.1 * max_y:
						
						colormap = plt.cm.get_cmap('tab10')
						color    = colormap((index % 10) / 10)
						
						spike_start_time = simulation_start_time + start * time_increment 
						spike_mid_time   = spike_start_time + duration / 2
						spike_mid_point  = (np.min(spike_values) + np.max(spike_values)) / 2
						
						# When the spike appears wider on the graph, then the text should be horizontal,
						# but if the spike appears taller, then it should be vertical. Note that we are
						# talking about how it appears *visually* on the plot, not how tall/wide it is numerically.
						spike_width_on_graph  = duration / simulation_total_time * (16/9) # Hardcoded 16/9 aspect ratio.
						spike_height_on_graph = mean_height / num_probes
						if spike_height_on_graph > spike_width_on_graph:
							rotation = 'vertical'
						else:
							rotation = 'horizontal'

						# The goal is to put the text against as white of a background as possible so that it's readable,
						# but the secondary goal is to cover up as little of the data as possible.

						# The default bounding box for the text has a ton of padding, so it would cover up a lot of
						# the measurements, that's why we make a custom bounding box with no padding. I also make it 
						# somewhat transparent so that you can still kind of make out the data underneath.
						bounding_box = dict(boxstyle='square', pad=0, ec=(1,1,1,0.8), fc=(1,1,1,0.8))
						ax[i].text(spike_mid_time, spike_mid_point, label, size=LABEL_SIZE, ha='center', va='center', color=color, backgroundcolor='w', rotation=rotation, bbox=bounding_box)


		# The clock's values should never be outside this range, and this way it will look a bit cleaner.
		if label == 'Clock':
			ax[i].set_ylim(-1.5, +1.5)

	ax[-1].set_xlabel('time')
	#ax[0].set_title(title, y=2)
	ax[0].set_title(title)
	#plt.tight_layout()
	fig.subplots_adjust(right=0.85)
	plt.show()