from autobound.causalProblem import causalProblem
from autobound.DAG import DAG
import io


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
