import numpy as np
from nengo import *
from nengo_spa import *
from utils import *

D = 32              # Number of dimensions for each ensemble.
N = 64              # Number of neurons per dimension.
CLOCK_PERIOD = 0.25 # How many seconds a full clock cycle takes.
SIM_TIME   = 100    # How long to run the simulation.
LEARN_TIME = 75     # How long until learning should be switched off.
SEED = 42           # RNG seed for deterministic results.

RNG = np.random.RandomState(SEED)
goals   = 'Start Increment Update Answer Done'.split()
actions = 'Yes No'.split()
numbers = 'Nil One Two Three Four'.split() # "Zero" is a built-in pointer, so that's why we use "Nil" instead.
letters = 'A B C D E F G H'.split()
concepts = letters + numbers

vocab_concepts = Vocabulary(dimensions=D, pointer_gen=RNG, max_similarity=0.01) # This is kind of necessary to get good SPs.
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
		count     = WTAAssocMem(threshold=0.3, input_vocab=vocab_concepts, mapping=vocab_concepts.keys(), function=lambda x: x > 0, n_neurons=N)
		letter    = WTAAssocMem(threshold=0.3, input_vocab=vocab_concepts, mapping=vocab_concepts.keys(), function=lambda x: x > 0, n_neurons=N)
		motor     = WTAAssocMem(threshold=0.3, input_vocab=vocab_motor, mapping=vocab_motor.keys(), function=lambda x: x > 0, n_neurons=N)

		Connection(goal.output,      goal.input,      synapse=0.05, transform=0.5)
		Connection(next_goal.output, next_goal.input, synapse=0.05, transform=0.5)
		Connection(count.output,     count.input,     synapse=0.05, transform=0.5)
		Connection(letter.output,    letter.input,    synapse=0.05, transform=0.5)
		Connection(motor.output,     motor.input,     synapse=0.05, transform=0.5)

	with Network('Clock'):
		clock      = Scalar(n_neurons=100)
		clock_node = Node(lambda t: +1 if t % CLOCK_PERIOD < CLOCK_PERIOD/2 else -1)
		Connection(clock_node, clock.input, synapse=None)

	with Network('Increment'):
		number_increment = WTAAssocMem(threshold=0.2, input_vocab=vocab_concepts, mapping=concept_mapping, function=lambda x: x > 0.0, n_neurons=N)
		Connection(number_increment.input,  number_increment.input,  synapse=0.05, transform=0.8)
		Connection(number_increment.output, number_increment.output, synapse=0.05, transform=0.8)

		letter_increment = WTAAssocMem(threshold=0.2, input_vocab=vocab_concepts, mapping=concept_mapping, function=lambda x: x > 0.0, n_neurons=N)
		Connection(letter_increment.input,  letter_increment.input,  synapse=0.05, transform=0.8)
		Connection(letter_increment.output, letter_increment.output, synapse=0.05, transform=0.8)

	with Network('Declarative'):
		#
		# probe  ---->  learn_in  --PES-->  learn_out  --Cleanup-->  retrieval 
		#                            ^
		#                            |
		#                          error = learn_out - letter              
		#
		probe = State(vocab=vocab_concepts, neurons_per_dimension=N)
		learn_in  = Ensemble(dimensions=D, n_neurons=D*N, intercepts=[0.5]*(D*N))
		learn_out = Ensemble(dimensions=D, n_neurons=D*N)
		retrieval = WTAAssocMem(threshold=0.2, input_vocab=vocab_concepts, mapping=vocab_concepts.keys(), function=lambda x: x > 0, n_neurons=N)
		
		Connection(probe.output, learn_in, synapse=None)
		pes = Connection(learn_in, learn_out, synapse=None, learning_rule_type=PES(learning_rate=0.1))
		Connection(learn_out, retrieval.input, synapse=None)
		
		error = Ensemble(dimensions=D, n_neurons=D*N)
		Connection(learn_out, error, synapse=None)
		Connection(letter.output, error, transform=-1, synapse=None)
		Connection(error, pes.learning_rule, synapse=None)
		
		learning   = Scalar()
		inhibition = Scalar()
		Connection(learning.output, learning.input)
		Connection(learning.output, inhibition.input, function=lambda x: 0 if x > 0.5 else -1)
		Connection(inhibition.output, error.neurons, transform=10*np.ones((error.n_neurons, 1)), synapse=None)

		retrieval_status = Scalar()
		Connection(Node(lambda t: -1 if t < LEARN_TIME else +1), retrieval_status.input)

	previously_done = True
	problem_index = 0
	def get_problem(t, next_input_signal):
		problems = [
			('A', 'Two',   'C'),
			('B', 'Two',   'D'),
			('C', 'Three', 'F'),
			('D', 'Three', 'G'),
			('A', 'Four',  'E'),
			('B', 'Four',  'F'),
			('C', 'Two',   'E'),
			('D', 'Two',   'F'),
			('A', 'Three', 'D'),
			('B', 'Three', 'E'),
			('C', 'Four',  'G'),
			('D', 'Four',  'H'),
		]
		global previously_done, problem_index
		if not previously_done and next_input_signal > 0.8:
			problem_index = (problem_index + 1) % len(problems)
			previously_done = True
			print('\ntime: %.03f' % t)
		elif next_input_signal < -0.8:
			previously_done = False
		return problems[problem_index]

	next_input_signal = Scalar()
	Transcode(lambda t: 'Start' if t < 0.1 else '0', output_vocab=vocab_goals) >> goal
	input_lhs    = Transcode(lambda t, n: get_problem(t, n)[0], size_in=1, output_vocab=vocab_concepts)
	input_rhs    = Transcode(lambda t, n: get_problem(t, n)[1], size_in=1, output_vocab=vocab_concepts)
	input_target = Transcode(lambda t, n: get_problem(t, n)[2], size_in=1, output_vocab=vocab_concepts)
	Connection(next_input_signal.output, input_lhs.input)
	Connection(next_input_signal.output, input_rhs.input)
	Connection(next_input_signal.output, input_target.input)
	Connection(next_input_signal.output, next_input_signal.input)
	input_lhs    >> lhs
	input_rhs    >> rhs
	input_target >> target
	lhs + rhs    >> probe

	with ActionSelection() as action_selection:

		goal_start     = dot(goal, sym.Start)
		goal_increment = dot(goal, sym.Increment)
		goal_update    = dot(goal, sym.Update)
		goal_answer    = dot(goal, sym.Answer)
		goal_done      = dot(goal, sym.Done)

		ifmax('Start',
			(1/2)*(goal_start + clock),
				-1      >> next_input_signal,
				-1      >> learning,
				sym.Nil >> count,
				lhs     >> letter,
				sym.Increment >> next_goal)

		ifmax('Retrieve',
			(1/3)*(goal_increment + retrieval_status + clock),
				rhs        >> compare_a,
				rhs        >> compare_b,
				retrieval  >> letter,
				sym.Update >> next_goal)

		ifmax('Increment',
			(1/3)*(goal_increment - retrieval_status + clock),
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
				+1     >> learning,
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
				+1 >> next_input_signal,
				-1 >> learning,
				sym.Start >> next_goal)

		ifmax('Update goal',
			-clock,
				next_goal >> goal)

	MyProbe(lhs.output,                 'LHS',         synapse=0.03, vocab=vocab_concepts)
	MyProbe(rhs.output,                 'RHS',         synapse=0.03, vocab=vocab_concepts)
	MyProbe(target.output,              'Target',      synapse=0.03, vocab=vocab_concepts)
	#MyProbe(goal.output,                'Goal',        synapse=0.03, vocab=vocab_goals)
	#MyProbe(next_goal.output,           'Next',        synapse=0.03, vocab=vocab_goals)
	#MyProbe(count.output,               'Count',       synapse=0.03, vocab=vocab_concepts)
	MyProbe(letter.output,              'Letter',      synapse=0.03, vocab=vocab_concepts)
	#MyProbe(compare_a.input,            'Compare A',   synapse=0.03, vocab=vocab_concepts)
	#MyProbe(compare_b.input,            'Compare B',   synapse=0.03, vocab=vocab_concepts)
	#MyProbe(comparison_status.output,   'CMP status',  synapse=0.03)
	#MyProbe(error,                      'Error',       synapse=0.03)
	MyProbe(retrieval.output,           'Retrieval',   synapse=0.03, vocab=vocab_concepts)
	#MyProbe(learn_out,                  'Learn out',   synapse=0.03, vocab=vocab_concepts)
	#MyProbe(learn_in,                   'Learn in',    synapse=0.03, vocab=vocab_concepts)
	#MyProbe(probe.output,               'Probe',       synapse=0.03, vocab=vocab_concepts)
	MyProbe(retrieval_status.output,    'Status',      synapse=0.03)
	#MyProbe(learning.output,            'Learning',    synapse=0.03)
	#MyProbe(inhibition.output,          'Inhibition',  synapse=0.03)
	MyProbe(motor.output,               'Answer',      synapse=0.03, vocab=vocab_motor)
	#MyProbe(next_input_signal.output,   'Next Input',  synapse=0.03)
	#MyProbe(clock.output,               'Clock',       synapse=0.01)

print('%d neurons' % model.n_neurons)
with Simulator(model) as sim:
	sim.run(SIM_TIME)
	
plot_simulation_output('Combined model', sim)