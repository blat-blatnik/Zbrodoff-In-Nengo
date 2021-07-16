import numpy as np
from nengo import *
from nengo_spa import *
from utils import *

D = 16              # Number of dimensions for each ensemble.
N = 64              # Number of neurons per dimension.
CLOCK_PERIOD = 0.25 # How many seconds a full clock cycle takes.
SEED = 42           # RNG seed for deterministic results.
SIM_TIME = 5        # How long to run the simulation (5 seconds should be plenty for clock period of 0.25).

RNG = np.random.RandomState(SEED)
problem = ['A', 'Three', 'D']

goals   = 'Start Increment Update Answer Done'.split()
actions = 'Yes No'.split()
numbers = 'Nil One Two Three Four'.split() # "Zero" is a built-in pointer, so that's why we use "Nil" instead.
letters = 'A B C D E F G H'.split()
concepts = letters + numbers

vocab_concepts = Vocabulary(dimensions=D, pointer_gen=RNG)
vocab_goals    = Vocabulary(dimensions=D, pointer_gen=RNG)
vocab_motor    = Vocabulary(dimensions=1, pointer_gen=RNG)
vocab_concepts.populate(';'.join(concepts))
vocab_goals.populate(';'.join(goals))
vocab_motor.populate(';'.join(actions))

number_mapping  = { numbers[i] : numbers[i + 1] for i in range(len(numbers) - 1) }
letter_mapping  = { letters[i] : letters[i + 1] for i in range(len(letters) - 1) }
concept_mapping = { **number_mapping, **letter_mapping }

with Network('Counting', seed=SEED) as model:

	with Network('Comparison'):
		compare_a = WTAAssocMem(threshold=0.5, input_vocab=vocab_concepts, mapping=vocab_concepts.keys(), function=lambda x: x > 0, n_neurons=N)
		compare_b = WTAAssocMem(threshold=0.5, input_vocab=vocab_concepts, mapping=vocab_concepts.keys(), function=lambda x: x > 0, n_neurons=N)	
		Connection(compare_a.output, compare_a.input, synapse=0.05, transform=0.8)
		Connection(compare_b.output, compare_b.input, synapse=0.05, transform=0.8)

		compare = Compare(vocab=vocab_concepts, neurons_per_dimension=N)
		Connection(compare_a.output, compare.input_a)
		Connection(compare_b.output, compare.input_b)

		# compare.output is a Node, and we can't make Connections with functions from a Node. So we need to pass it through a Scalar first.
		compare_output = Scalar()
		Connection(compare.output, compare_output.input, synapse=None)

		comparison_status = Scalar()
		Connection(compare_output.output, comparison_status.input, function=lambda x: +1 if x > 0.5 else -1)

	with Network('Goal'):
		lhs    = State(vocab=vocab_concepts)
		rhs    = State(vocab=vocab_concepts)
		target = State(vocab=vocab_concepts)

	with Network('Working Memory'):
		goal      = WTAAssocMem(threshold=0.5, input_vocab=vocab_goals, mapping=vocab_goals.keys(), function=lambda x: x > 0, n_neurons=N)
		next_goal = WTAAssocMem(threshold=0.5, input_vocab=vocab_goals, mapping=vocab_goals.keys(), function=lambda x: x > 0, n_neurons=N)
		count     = WTAAssocMem(threshold=0.5, input_vocab=vocab_concepts, mapping=vocab_concepts.keys(), function=lambda x: x > 0, n_neurons=N)
		letter    = WTAAssocMem(threshold=0.5, input_vocab=vocab_concepts, mapping=vocab_concepts.keys(), function=lambda x: x > 0, n_neurons=N)
		motor     = WTAAssocMem(threshold=0.5, input_vocab=vocab_motor, mapping=vocab_motor.keys(), function=lambda x: x > 0, n_neurons=N)

		Connection(goal.output,      goal.input,      synapse=0.05, transform=0.8)
		Connection(next_goal.output, next_goal.input, synapse=0.05, transform=0.8)
		Connection(count.output,     count.input,     synapse=0.05, transform=0.8)
		Connection(letter.output,    letter.input,    synapse=0.05, transform=0.8)
		Connection(motor.output,     motor.input,     synapse=0.05, transform=0.8)

	with Network('Clock'):
		clock      = Scalar(n_neurons=100)
		clock_node = Node(lambda t: +1 if t % CLOCK_PERIOD < CLOCK_PERIOD/2 else -1)
		#clock_node = Node(lambda t: np.sin(t * 2*np.pi/CLOCK_PERIOD))
		Connection(clock_node, clock.input, synapse=None)

	with Network('Increment'):
		number_increment = WTAAssocMem(threshold=0.5, input_vocab=vocab_concepts, mapping=concept_mapping, function=lambda x: x > 0.0, n_neurons=N)
		letter_increment = WTAAssocMem(threshold=0.5, input_vocab=vocab_concepts, mapping=concept_mapping, function=lambda x: x > 0.0, n_neurons=N)

		Connection(number_increment.input, number_increment.input, synapse=0.05, transform=0.8)
		Connection(letter_increment.input, letter_increment.input, synapse=0.05, transform=0.8)

	# This circuit is only used to time precisely how long everything took.
	is_done = False
	def timer_func(t, done):
		global is_done
		if not is_done and done > 0.8:
			is_done = True
			print('\ntime: %.03f' % t)
		return '0'
	timer = Transcode(timer_func, size_in=1, output_vocab=vocab_concepts)
	done_signal = Scalar()
	Connection(done_signal.output, timer.input)

	Transcode(lambda t: 'Start' if t < CLOCK_PERIOD/2 else '0', output_vocab=vocab_goals) >> goal
	Transcode(lambda t: problem[0], output_vocab=vocab_concepts) >> lhs
	Transcode(lambda t: problem[1], output_vocab=vocab_concepts) >> rhs
	Transcode(lambda t: problem[2], output_vocab=vocab_concepts) >> target

	with ActionSelection() as action_selection:

		goal_start     = dot(goal, sym.Start)
		goal_increment = dot(goal, sym.Increment)
		goal_update    = dot(goal, sym.Update)
		goal_answer    = dot(goal, sym.Answer)
		goal_done      = dot(goal, sym.Done)

		ifmax('Start',
			(1/2)*(goal_start + clock),
				-1      >> done_signal,
				sym.Nil >> count,
				lhs     >> letter,
				sym.Increment >> next_goal)

		ifmax('Increment',
			(1/2)*(goal_increment + clock),
				count  >> compare_a,
				rhs    >> compare_b,
				count  >> number_increment,
				letter >> letter_increment,
				sym.Update >> next_goal)

		ifmax('Update',
			(1/3)*(goal_update - comparison_status + clock),
				number_increment >> count,
				letter_increment >> letter,
				sym.Increment >> next_goal)
				
		ifmax('Compare',
			(1/3)*(goal_update + comparison_status + clock),
				letter >> compare_a,
				target >> compare_b,
				sym.Answer >> next_goal)

		ifmax('Answer Yes',
			(1/3)*(goal_answer + comparison_status + clock),
				sym.Yes  >> motor,
				sym.Done >> next_goal)

		ifmax('Answer No',
			(1/3)*(goal_answer - comparison_status + clock),
				sym.No   >> motor,
				sym.Done >> next_goal)

		ifmax('Done', 
			(1/2)*(goal_done + clock),
				+1 >> done_signal,
				sym.Done >> next_goal)

		ifmax('Update goal',
			-clock,
				next_goal >> goal)

	MyProbe(count.output,                     'Count',       synapse=0.03, vocab=vocab_concepts)
	MyProbe(letter.output,                    'Letter',      synapse=0.03, vocab=vocab_concepts)
	#MyProbe(compare_a.input,                  'Compare A',   synapse=0.03, vocab=vocab_concepts)
	#MyProbe(compare_b.input,                  'Compare B',   synapse=0.03, vocab=vocab_concepts)
	#MyProbe(comparison_status.output,         'CMP status',  synapse=0.03)
	MyProbe(motor.output,                     'Answer',      synapse=0.03, vocab=vocab_motor)
	MyProbe(goal.output,                      'Goal',        synapse=0.03, vocab=vocab_goals)
	#MyProbe(next_goal.output,                 'Next',        synapse=0.03, vocab=vocab_goals)
	#MyProbe(action_selection.thalamus.output, 'Actions',     synapse=0.01, mapping=action_selection.keys())
	#MyProbe(action_selection.bg.input,        'Utility',     synapse=0.01, mapping=action_selection.keys())
	MyProbe(clock.output,                     'Clock',       synapse=0.01)

print('%d neurons' % model.n_neurons)
with Simulator(model) as sim:
	sim.run(SIM_TIME)
	
plot_simulation_output('%s + %s = %s' % (problem[0], problem[1], problem[2]), sim)