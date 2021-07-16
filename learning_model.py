import numpy as np
from nengo import *
from nengo_spa import *
from utils import *

D = 16              # Number of dimensions for each ensemble.
N = 64              # Number of neurons per dimension.
CLOCK_PERIOD = 0.25 # How many seconds a full clock cycle takes.
SIM_TIME   = 20     # How long to run the simulation.
LEARN_TIME = 15     # How long until learning should be switched off.
SEED = 42           # RNG seed for deterministic results.

RNG = np.random.RandomState(SEED)
goals   = 'Start Retrieve Compare'.split()
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

ideal_mapping = {}
for number_idx in range(len(numbers)):
	for letter_idx in range(len(letters) - number_idx):
		probe  = '%s + %s' % (letters[letter_idx], numbers[number_idx])
		ideal_mapping[probe] = letters[letter_idx + number_idx]

with Network('Counting', seed=SEED) as model:

	with Network('Comparison'):
		compare_a = WTAAssocMem(threshold=0.3, input_vocab=vocab_concepts, mapping=vocab_concepts.keys(), function=lambda x: x > 0, n_neurons=N)
		compare_b = WTAAssocMem(threshold=0.3, input_vocab=vocab_concepts, mapping=vocab_concepts.keys(), function=lambda x: x > 0, n_neurons=N)	
		Connection(compare_a.output, compare_a.input, synapse=0.05, transform=0.8)
		Connection(compare_b.output, compare_b.input, synapse=0.05, transform=0.8)

		compare = Compare(vocab=vocab_concepts, neurons_per_dimension=N)
		Connection(compare_a.output, compare.input_a)
		Connection(compare_b.output, compare.input_b)

		# compare.output is a Node, and we can't make Connections with functions from a Node. So we need to pass it through a Scalar first.
		compare_output = Scalar()
		Connection(compare.output, compare_output.input, synapse=None)

		comparison_status = Scalar()
		Connection(comparison_status.output, comparison_status.input, synapse=0.05)
		Connection(compare_output.output, comparison_status.input, function=lambda x: 1 if x > 0.5 else -1)

	with Network('Goal'):
		lhs    = State(vocab=vocab_concepts)
		rhs    = State(vocab=vocab_concepts)
		target = State(vocab=vocab_concepts)

	with Network('Imaginal'):
		goal      = WTAAssocMem(threshold=0.3, input_vocab=vocab_goals, mapping=vocab_goals.keys(), function=lambda x: x > 0, n_neurons=N)
		next_goal = WTAAssocMem(threshold=0.3, input_vocab=vocab_goals, mapping=vocab_goals.keys(), function=lambda x: x > 0, n_neurons=N)
		motor     = WTAAssocMem(threshold=0.3, input_vocab=vocab_motor, mapping=vocab_motor.keys(), function=lambda x: x > 0, n_neurons=N)

		Connection(goal.output,      goal.input,      synapse=0.05, transform=0.5)
		Connection(next_goal.output, next_goal.input, synapse=0.05, transform=0.5)
		Connection(motor.output,     motor.input,     synapse=0.05, transform=0.5)

	with Network('Declarative'):
		ideal_memory = WTAAssocMem(threshold=0.5, input_vocab=vocab_concepts, mapping=ideal_mapping, function=lambda x: x > 0, n_neurons=N) 
		
		#
		# probe  ---->  learn_in  --PES-->  learn_out  --Cleanup-->  retrieval 
		#                            ^
		#                            |
		#                          error = learn_out - ideal_memory              
		#
		probe = State(vocab=vocab_concepts, neurons_per_dimension=N)
		learn_in  = Ensemble(dimensions=D, n_neurons=D*N, intercepts=[0.5]*(D*N)) # 0.4 might be better.
		learn_out = Ensemble(dimensions=D, n_neurons=D*N)
		retrieval = WTAAssocMem(threshold=0.5, input_vocab=vocab_concepts, mapping=vocab_concepts.keys(), function=lambda x: x > 0, n_neurons=N)
		
		Connection(probe.output, learn_in, synapse=None)
		pes = Connection(learn_in, learn_out, synapse=None, learning_rule_type=PES(learning_rate=1e-3))
		Connection(learn_out, retrieval.input, synapse=None)
		
		error = Ensemble(dimensions=D, n_neurons=D*N)
		Connection(learn_out, error, synapse=None)
		Connection(ideal_memory.output, error, transform=-1, synapse=None)
		Connection(error, pes.learning_rule, synapse=None)
		
		inhibition = Node(lambda t: 0 if t < LEARN_TIME else -10)
		Connection(inhibition, error.neurons, transform=np.ones((error.n_neurons, 1)), synapse=None)

	with Network('Clock'):
		clock      = Scalar(n_neurons=100)
		clock_node = Node(lambda t: +1 if t % CLOCK_PERIOD < CLOCK_PERIOD/2 else -1)
		Connection(clock_node, clock.input, synapse=None)

	def get_problem(t):
		cycle = int(t / CLOCK_PERIOD)
		index = cycle // len(goals)
		problems = [
			('A', 'Two', 'C'),
			('B', 'Two', 'D'),
			('C', 'Three', 'F'),
			('D', 'Three', 'F'),
			('A', 'Four', 'D'),
			('B', 'Four', 'G'),
		]
		return problems[index % len(problems)]

	Transcode(lambda t: 'Start' if t < CLOCK_PERIOD/2 else '0', output_vocab=vocab_goals) >> goal
	Transcode(lambda t: get_problem(t)[0], output_vocab=vocab_concepts) >> lhs
	Transcode(lambda t: get_problem(t)[1], output_vocab=vocab_concepts) >> rhs
	Transcode(lambda t: get_problem(t)[2], output_vocab=vocab_concepts) >> target
	lhs + rhs >> ideal_memory
	lhs + rhs >> probe

	with ActionSelection() as action_selection:

		goal_start    = dot(goal, sym.Start)
		goal_retrieve = dot(goal, sym.Retrieve)
		goal_compare  = dot(goal, sym.Compare)

		# Technically we don't need this many steps (only the compare is needed).
		# Hopefully this should make it easier to integrate with the counting model though.
		ifmax('Start',
			(1/2)*(goal_start + clock),
				sym.Retrieve >> next_goal)

		ifmax('Retrieve',
			(1/2)*(goal_retrieve + clock),
				retrieval   >> compare_a,
				target      >> compare_b,
				sym.Compare >> next_goal)
				
		ifmax('Compare Yes',
			(1/3)*(goal_compare + comparison_status + clock),
				sym.Yes   >> motor,
				sym.Start >> next_goal)

		ifmax('Compare No',
			(1/3)*(goal_compare - comparison_status + clock),
				sym.No    >> motor,
				sym.Start >> next_goal)

		ifmax('Update goal',
			-clock,
				next_goal >> goal)

	MyProbe(lhs.output,               'LHS',         synapse=0.03, vocab=vocab_concepts)
	MyProbe(rhs.output,               'RHS',         synapse=0.03, vocab=vocab_concepts)
	#MyProbe(target.output,            'target',      synapse=0.03, vocab=vocab_concepts)
	#MyProbe(learn_in,                 'Learn in',  synapse=0.03, vocab=vocab_concepts)
	#MyProbe(learn_out,                'Learn out', synapse=0.03, vocab=vocab_concepts)
	MyProbe(ideal_memory.output,      'Ideal Mem.',  synapse=0.03, vocab=vocab_concepts) 
	MyProbe(retrieval.output,         'Retrieval',   synapse=0.03, vocab=vocab_concepts) 
	#MyProbe(error,                    'Error',       synapse=0.03)
	#MyProbe(goal.output,              'Goal',        synapse=0.03, vocab=vocab_goals)
	#MyProbe(next_goal.output,         'Next',        synapse=0.03, vocab=vocab_goals)
	#MyProbe(compare_a.input,          'Compare A',   synapse=0.03, vocab=vocab_concepts)
	#MyProbe(compare_b.input,          'Compare B',   synapse=0.03, vocab=vocab_concepts)
	#MyProbe(comparison_status.output, 'CMP status',  synapse=0.03)
	MyProbe(inhibition,               'Inhibition',  synapse=0.03)
	#MyProbe(motor.output,             'Motor',       synapse=0.03, vocab=vocab_motor)
	#MyProbe(clock.output,             'Clock')

print('%d neurons' % model.n_neurons)
with Simulator(model) as sim:
	sim.run(SIM_TIME)
	
plot_simulation_output('Learning model', sim)