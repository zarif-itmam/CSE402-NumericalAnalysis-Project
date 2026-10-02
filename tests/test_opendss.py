import opendssdirect as dss

# Initialize and create a minimal test circuit
dss.run_command("Clear")
dss.run_command("New Circuit.TestCircuit basekv=115 pu=1.0 phases=3 bus1=SourceBus")

# Run power flow solution
dss.Solution.Solve()

# Verify convergence and retrieve results
if dss.Solution.Converged():
    print("OpenDSSDirect setup successful! Solution converged.")
    print("Circuit Name:", dss.Circuit.Name())
    print("System Bus Names:", dss.Circuit.AllBusNames())
else:
    print("OpenDSS initialized, but the test circuit failed to converge.")