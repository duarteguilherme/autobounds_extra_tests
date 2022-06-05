from autobound.causalProblem import causalProblem
from autobound.DAG import DAG
import io


def test_selection():
    dag = DAG()
    dag.from_structure("Y -> S, X -> Y, U -> X, U -> Y", unob = "U")
    problem = causalProblem(dag)
    problem.load_data('data/selection_obsqty.csv')
    problem.set_estimand(problem.query('Y(X=1)=1') + problem.query('Y(X=0)=1', -1))
    problem.add_prob_constraints()
    program = problem.write_program()
    result = program.run_pyomo('ipopt')
    assert result[0] <= -0.49
    assert result[0] >= -0.51
    assert result[1] <= 0.65
    assert result[1] >= 0.63

def test_manski2():
    dag = DAG()
    dag.from_structure("X -> Y, X -> R, R -> S, Y -> S")
    problem = causalProblem(dag)
    problem.load_data('data/manski2_1.csv', cond = ['R','S'])
    problem.load_data('data/manski2_2.csv', cond = ['R'])
    problem.load_data('data/manski2_3.csv', cond = ['S'])
    problem.load_data('data/manski2_4.csv')
    problem.set_estimand(problem.query('Y(X=1)=1') + problem.query('Y(X=0)=1', -1))
    problem.add_prob_constraints()
    program = problem.write_program()
    result = program.run_pyomo('ipopt')
    assert result[0] <= 0.5
    assert result[0] >= 0.49
    assert result[1] <= 0.5
    assert result[1] >= 0.49

def test_manski1():
    dag = DAG()
    dag.from_structure("X -> Y, Y -> R, X -> R")
    problem = causalProblem(dag)
    problem.load_data('data/manski1_1.csv', cond = ['R'])
    problem.load_data('data/manski1_2.csv')
    problem.set_estimand(problem.query('Y(X=1)=1') + problem.query('Y(X=0)=1', -1))
    problem.add_prob_constraints()
    program = problem.write_program()
    result = program.run_pyomo('ipopt')
    result
    assert result[0] <= -0.24
    assert result[0] >= -0.26
    assert result[1] <= 0.75
    assert result[1] >= 0.74

def test_measurement_error():
    dag = DAG()
    dag.from_structure("X -> Y, Y -> S, U -> S, U -> Y", unob = "U")
    problem = causalProblem(dag)
    problem.load_data('data/measurement_error.csv')
    problem.set_estimand(problem.query('Y(X=1)=1') + problem.query('Y(X=0)=1', -1))
    problem.add_constraint(problem.query('S(Y=0)=1&S(Y=1)=0'))
    problem.add_prob_constraints()
    program = problem.write_program()
    result = program.run_pyomo('ipopt')
    program.to_pip('/home/beta/measurement.pip')
    assert result[0] <= -0.6
    assert result[0] >= -0.64
    assert result[1] <= 1.01
    assert result[1] >= 0.98

def return_iv_problems():
    dag = DAG()
    dag.from_structure("Z -> X, X -> Y, U -> X, U -> Y", unob = "U")
    dag2 = DAG()
    dag2.from_structure("Z -> Y, Z -> X, X -> Y, U -> X, U -> Y", unob = "U")
    just_problem = causalProblem(dag)
    overly_cautious_problem = causalProblem(dag2)
    overconfident_problem = causalProblem(dag)
    overconfident_problem.set_p_to_zero(
            [x[1][0] for x in overconfident_problem.query('X(Z=0)=1&X(Z=1)=0')]
            )
    just_problem.load_data('data/iv.csv')
    just_problem.add_prob_constraints()
    overly_cautious_problem.load_data('data/iv.csv')
    overly_cautious_problem.add_prob_constraints()
    overconfident_problem.load_data('data/iv.csv')
    overconfident_problem.add_prob_constraints()
    return (just_problem, overly_cautious_problem,overconfident_problem)

def test_program_iv_ate():
    p1ate, p2ate, p3ate = return_iv_problems()
    p1ate.set_estimand(p1ate.query('Y(X=1)=1') + p1ate.query('Y(X=0)=1', -1))
    p2ate.set_estimand(p2ate.query('Y(X=1)=1') + p2ate.query('Y(X=0)=1', -1))
    p3ate.set_estimand(p3ate.query('Y(X=1)=1') + p3ate.query('Y(X=0)=1', -1))
    p1ate, p2ate, p3ate = p1ate.write_program(), p2ate.write_program(), p3ate.write_program()
    p1_ate_result = p1ate.run_pyomo('ipopt')
    p2_ate_result = p2ate.run_pyomo('ipopt')
    p3_ate_result = p3ate.run_pyomo('ipopt')
    p2_ate_result
    assert p1_ate_result[0] <= -0.54
    assert p1_ate_result[0] >= -0.56
    assert p1_ate_result[1] <= -0.13
    assert p1_ate_result[1] >= -0.15
    assert p2_ate_result[0] <= -0.62
    assert p2_ate_result[0] >= -0.63
    assert p2_ate_result[1] <= 0.38
    assert p2_ate_result[1] >= 0.36

def test_program_iv_late():
    p1late, p2late, p3late = return_iv_problems()
    p1late.set_estimand(
            p1late.query('Y(X=1)=1&X(Z=1)=1&X(Z=0)=0') + 
            p1late.query('Y(X=0)=1&X(Z=1)=1&X(Z=0)=0', -1),
            div = p1late.query('X(Z=1)=1&X(Z=0)=0'))
    p2late.set_estimand(
            p2late.query('Y(X=1)=1&X(Z=1)=1&X(Z=0)=0') + 
            p2late.query('Y(X=0)=1&X(Z=1)=1&X(Z=0)=0', -1),
            div = p2late.query('X(Z=1)=1&X(Z=0)=0'))
    p3late.set_estimand(
            p3late.query('Y(X=1)=1&X(Z=1)=1&X(Z=0)=0') + 
            p3late.query('Y(X=0)=1&X(Z=1)=1&X(Z=0)=0', -1),
            div = p3late.query('X(Z=1)=1&X(Z=0)=0'))
    p1late, p2late, p3late = p1late.write_program(), p2late.write_program(), p3late.write_program()
    p1_late_result = p1late.run_pyomo('ipopt')
    p2_late_result = p2late.run_pyomo('ipopt')
    p3_late_result = p3late.run_pyomo('ipopt')
    assert p1_late_result[0] <= -0.99
    assert p1_late_result[1] >= 0.99
    assert p2_late_result[0] <= -0.99
    assert p2_late_result[1] >= 0.99
