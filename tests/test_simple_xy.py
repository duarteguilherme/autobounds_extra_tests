# Testing conditional data
# Introducing Data on P(Y|X) First
# Model is X -> Y, no confounding
# Then tests Model X -> Y with confounding
# But data is introduced cond + single X



from autobound.causalProblem import causalProblem
from autobound.DAG import DAG
import io


def solve_cond_xy_no_conf():
    dag = DAG()
    dag.from_structure("X -> Y")
    problem = causalProblem(dag)
    problem.load_data('data/simple_xy.csv', cond = ['X'])
    problem.load_data('data/simple_xy_only_x.csv')
    problem.add_prob_constraints()
    problem.set_estimand(problem.query('Y(X=1)=1') + problem.query('Y(X=0)=1', -1))
    program = problem.write_program()
    result = program.run_pyomo('couenne')
    program.to_pip('/home/beta/simple_xy.pip')
    assert result[0] >= 0.39
    assert result[0] <= 0.41
    assert result[1] >= 0.39
    assert result[1] <= 0.41
