
from ax import *

branin_search_space = SearchSpace(
    parameters=[
        RangeParameter(
            name="x1", parameter_type=ParameterType.FLOAT, lower=-5, upper=10
        ),
        RangeParameter(
            name="x2", parameter_type=ParameterType.FLOAT, lower=0, upper=15
        ),
    ]
)
exp = SimpleExperiment(
    name="test_branin",
    search_space=branin_search_space,
    evaluation_function=lambda p: branin(p["x1"], p["x2"]),
    objective_name="branin",
    minimize=True,
)

sobol = Models.SOBOL(exp.search_space)
for i in range(5):
    exp.new_trial(generator_run=sobol.gen(1))

best_arm = None
for i in range(15):
    gpei = Models.GPEI(experiment=exp, data=exp.eval())
    generator_run = gpei.gen(1)
    best_arm, _ = generator_run.best_arm_predictions
    exp.new_trial(generator_run=generator_run)

best_parameters = best_arm.parameters
print(best_parameters)
