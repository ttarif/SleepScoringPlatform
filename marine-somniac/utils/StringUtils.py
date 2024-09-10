class StringUtils:
    @staticmethod
    def extract_arg_desc_from_docstring(doc_str: str, arg: str) -> str|None:
        """
        The docstrings of EDF.Channel, .EXGChannel, and .ECGChannel follow a specific
        format such that the description of an argument can be extracted with the following
        string operation.
        """
        if doc_str is not None:
            if arg in doc_str:
                desc = doc_str.split(f"{arg}:")[1].split('\n')[0]
                return desc
        else:
            return None
        
    def get_channel_methods(self, ch_name) -> list:
        ch_type = self.edf.channel_types[ch_name]
        channel_obj = self.edf._route_object[ch_type]
        return [i for i in dir(channel_obj) if 'get' in i and '__' not in i]
    
    @staticmethod
    def format_method_arg_labels(argdict: dict) -> str:

        def remove_chain(base_str: str, replacements: list):
            removed = base_str
            for r in replacements:
                removed = removed.replace(r, '') 
            return removed
        
        remove = ["'", '"', '{', '}']
        label = f": ({remove_chain(str(argdict), remove)})"
        return label