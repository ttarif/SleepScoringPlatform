class Base:
    def run_method(self, method_name, args: dict):
        func = getattr(self, method_name) 
        return func(**args)