# src/test_flow.py
from metaflow import FlowSpec, step, kubernetes

class TestFlow(FlowSpec):

    @kubernetes
    @step
    def start(self):
        import sys
        print(f"✅ Python version in K8s container: {sys.version}")
        self.next(self.end)

    @kubernetes
    @step
    def end(self):
        print("✅ Done running on Kubernetes!")

if __name__ == "__main__":
    TestFlow()
