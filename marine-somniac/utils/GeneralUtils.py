import json
import typing

class GeneralUtils:
    @staticmethod
    def read_json(path) -> dict:
        with open(path) as f:
            json_dict = json.load(f)
        return json_dict
    
    @staticmethod
    def convert_type(supertype, value):
        if supertype is not None:
            if typing.get_args(supertype) == ():
                try:
                    return supertype(value)
                except:
                    raise TypeError(f"Could not convert `{value}` into {supertype}")
            else:
                collection = []
                subtypes = typing.get_args(supertype)

                if len(subtypes) == 1 and len(value) > 1 and typing.get_origin(supertype) is list:
                    subtypes = subtypes*len(value)
                for i, v in enumerate(value):
                    collection.append(GeneralUtils.convert_type(
                        subtypes[i], v
                    ))
                return typing.get_origin(supertype)(collection)