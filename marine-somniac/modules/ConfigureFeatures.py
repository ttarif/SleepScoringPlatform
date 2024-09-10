import streamlit as st
import pandas as pd
from typing import Callable, get_origin, get_args
import inspect
import modules.instructions as instruct
from utils.EDF.EDF import EDFutils
from utils.EDF.constants import HR_BANDS
from modules.ConfigureSession import SessionConfig
from config.channelcompute import *


class ConfigureFeatures(SessionConfig):
    def __init__(self, analysis) -> None:
        self.analysis = analysis
        self.output_freq = None
        self.edf = EDFutils(
            self.get_edf_from_analysis(self.analysis),
            fetch_metadata=False,
            config=self.get_edfconfig() 
        )
        self.feature_config = {}
        self.validities = {}
        self.config_name = "MakeFeatures.json"
        self.feature_data_name = "features.csv"
        self.containers = {}
        # bad design, forces all method names to be unique...
        self.name_to_method = {m.__name__: m for m in FEATURE_OPTIONS['all'].values()}

    # TODO
    def get_widget_defaults(self, preload_type):
        match preload_type:

            case 'From scratch':
                pass
        return 

    def specify_methods_per_channel(self) -> None:
        """
        Create expanders for each specified channel and let user specify which
        features should be computed from it. Modifies self.feature_config to contain
        all the specifications generated by the nested functions.
        """
        st.subheader("Specify first round calculations")
        channel_methods = {}
        n_ch = len(self.edf.channels)
        n = n_ch if n_ch < 3 else 3
        c = st.columns(n)
        for i, (ch_name, ch_type) in enumerate(self.edf.channel_types.items()):
            with c[i%n].expander(f"({ch_type}): {ch_name}", True):
                channel_methods[ch_name] = st.multiselect(
                    "Calculate features",
                    options=FEATURE_OPTIONS[ch_type],
                    default=FEATURE_OPTIONS[ch_type],
                    key=f"{ch_name}-pick-derivatives"
                )
                m_cfgs = []
                for method_label in channel_methods[ch_name]:
                    method = FEATURE_OPTIONS['all'][method_label]
                    method_config = self.specify_method_instances(ch_name, method_label, method)
                    m_cfgs.append(method_config)
                self.feature_config[ch_name] = {k: v for cfg in m_cfgs for k, v in cfg.items()}

                validity = self.validate_channel_configuration(ch_name, self.feature_config[ch_name])
                if not validity[0]:
                    st.error(validity[1])
                else:
                    st.success(validity[1])
                
    def specify_method_instances(self, ch_name:str, label:str, method:Callable, previous_derivand:list=None) -> dict:
        method_config = []
        method_name = method.__name__
        og_label = label

        if previous_derivand is None:
            previous_derivand = []

        # Is the method configurable?
        if label in NOT_CONFIGURABLE:
            method_config.append({'args': None})
        else:
            # Is this a recursive call?
            if previous_derivand:
                label = f"{' '.join(previous_derivand)}: {og_label}"
            popover = st.popover(f"Configure {label}", use_container_width=True)

            # Is the method derivable?
            if not og_label in DERIVANDS:
                main_menu = popover.container()
            else:
                main_menu, deriv_menu = popover.tabs([f"{label}", f'{label} Derivatives'])
            
            n_instances = main_menu.number_input("How many instances?",
                    min_value=1,
                    key=f"{ch_name}{label}",
                    help=instruct.N_COMPS_HELP
                )
            main_menu.markdown(f"**Parameters for {og_label} Calculation**")
            for i in range(n_instances):
                deriv_config = {}
                main_menu.markdown(f"Instance {i+1}")
                method_args = self.specify_method_arguments(
                    main_menu, ch_name, method, label+str(i))

                if og_label in DERIVANDS:
                    deriv_menu.markdown(f"**Instance {i+1}{self.format_method_arg_labels(method_args)}**")
                    derivatives = deriv_menu.multiselect(
                        "Calculate features",
                        options=DERIVANDS[og_label],
                        default=DERIVANDS[og_label],
                        key=f"{ch_name}{label}{i}"
                    )
                    deriv_config = {}
                    for deriv in derivatives:
                        current_derivand = [f"{label} {i+1}"]
                        next_method = LABEL_TO_METHOD[deriv]
                        # recursive call
                        
                        deriv_config = {**self.specify_method_instances(
                            ch_name=ch_name,
                            label=deriv,
                            method=next_method,
                            previous_derivand=previous_derivand+current_derivand,
                        ), **deriv_config}

                derived = {'derived': deriv_config} if deriv_config else {}
                method_config.append({
                    'args': method_args,
                    **derived
                })
        return {method_name: method_config}
    
    def specify_method_arguments(self, container, ch_name, method, key_str='') -> dict:
        """
        Generate the widgets that allow the user to modify the parameters
        going into a feature computation. Sets defaults and gives argument
        descriptions by reading the method argspec and docstring.
        """
        method_name = method.__name__
        arg_info = self.get_method_args(method, forspec=True)  
    
        arg_vals = {}
        unpacked_args = {}
        if not method_name in CUSTOM_ARGSPEC:
            c = container.columns(len(arg_info))
            for i, (arg_name, info) in enumerate(arg_info.items()):
                if not 'tuple_flag' in info:
                    arg_vals[arg_name] = c[i].text_input(
                        arg_name,
                        value=info['default'],
                        help=info['help'],
                        key=f"{key_str}{ch_name}{method_name}{arg_name}"
                    )
                else:
                    name, pos = info['tuple_flag']
                    if name not in unpacked_args:
                        unpacked_args[name] = [None, None]
                    unpacked_args[name][pos] = c[i].text_input(
                        arg_name,
                        value=info['default'],
                        help=info['help'],
                        key=f"{key_str}{ch_name}{method_name}{arg_name}"
                    )
            arg_vals = {
                **unpacked_args,
                **arg_vals
            }
        else:
            if method_name in ('get_welch', 'get_hr_welch'):
                wkey = key_str+ch_name+method_name
                arg_vals = self.specify_welch_args(container, arg_info, wkey)
            else:
                raise Exception("custom method flagged, but no process specified")

        return arg_vals
    
    def specify_welch_args(self, container, arg_info, key) -> dict:
        arg_vals = {}
        with container:
            arg_vals['window_sec'] = st.number_input(
                'window_sec',
                value=arg_info['window_sec']['default'],
                help=arg_info['window_sec']['help'],
                key=key+'window_sec'
            )
            bands = []
            for band in arg_info['bands']['default']:
                c = st.columns(3)
                name = c[0].text_input(
                    'band name',
                    value=band[2],
                    help="Name of the band range. If you are using the pre-populated defaults, "
                         "do not change this as it influences downstream calculations. "
                         "Erasing the name removes this calculation.",
                    key=key+'band_name'+str(band)
                )
                if name: 
                    low = c[1].text_input(
                        'range low end',
                        value=band[0],
                        help="Low end of frequency range for which to calculate power",
                        key=key+'low'+str(band)
                    )
                    high = c[2].text_input(
                        'range high end',
                        value=band[1],
                        help="High end of frequency range for which to calculate power",
                        key=key+'high'+str(band)
                    )
                bands.append((low, high, name))
            arg_vals['bands'] = [i for i in bands if i[2]]
        return arg_vals

    def validate_channel_configuration(self, ch_name, channel_config) -> tuple[bool, str]:
        if not channel_config:
            return (False, "No features specified. Either remove this channel from the main "
                    "configuration, or specify features to compute from this channel.")
        
        # try:
        typed_config = self.iterate_type_coercion(ch_name, channel_config)
        self.feature_config[ch_name] = typed_config
        self.validities[ch_name] = True
        return (True, "Configuration valid")
        # except Exception as exc:
        #     self.validities[ch_name] = False
        #     return (False, str(exc))

    def validate_all_configurations(self) -> tuple[bool, str]:
        if not len(self.validities) == len(self.edf.channels):
            return (False, "Please specify configurations for all channels")
        if not all(self.validities.values()):
            return (False, "One or more channels have invalid configurations.")
        return (True, "All configurations valid. "
                "Saving this configuration will overwrite the previous.")

    def iterate_type_coercion(self, ch_name: str, tree: dict) -> tuple|dict:
        converted = {}
        for method_name, instances in tree.items():
            # bad design, forces all method names to be unique...
            method = self.name_to_method[method_name]
            converted[method_name] = []
            arg_info = self.get_method_args(method)
            for i, instance in enumerate(instances):
                inst_config = {'args': {}}
                # args = None when non-configurable method like most Epoch-derived
                if instance['args'] is not None:
                    for arg, value in instance['args'].items():
                        arg_type = arg_info[arg]['type']
                        typed_val = self.convert_type(arg_type, value)
                        inst_config['args'][arg] = typed_val
                    if instance.get('derived'):
                        dtree = tree[method_name][i]['derived']
                        inst_config['derived'] = self.iterate_type_coercion(ch_name, dtree)
                    converted[method_name].append(inst_config)
        return converted
    
    # TODO
    def retrieve_configuration(self) -> dict:
        cfg_path = self.get_file_from_analysis(self.config_name)
        if cfg_path is not None:
            return self.read_json(cfg_path)

    def save_configuration(self) -> None:
        self.write_configuration(
            analysis=self.analysis,
            config=self.feature_config,
            name=self.config_name
        )
        st.toast(f"Configuration saved.")

    # TODO
    def build_features(self):
        with st.spinner("Calculating features, this may take a while..."):
            pass
            # df.to_csv(f"{}/{self.feature_data_name}")
        st.toast("Features computed and saved to analysis.")

    def get_method_args(self, method: Callable, forspec=False) -> dict:
        argspec = inspect.getfullargspec(method)

        args = [arg for arg in argspec.args if arg != 'self']
        arg_defaults = argspec.defaults
        if arg_defaults is None:
            arg_defaults = [None]*len(args)
        arg_descs = [self.extract_arg_desc_from_docstring(method.__doc__, arg) for arg in args]
        arg_types = [argspec.annotations.get(arg) for arg in args]

        arg_info = {}
        for i, arg in enumerate(args):
            is_special_tuple = get_origin(arg_types[i]) is tuple and \
                                len(get_args(arg_types[i])) == 2 and \
                                isinstance(arg_defaults[i], tuple)
            if forspec and is_special_tuple:
                ltype, htype = get_args(arg_types[i])
                ldef, hdef = arg_defaults[i]
                arg_info[f'low_{arg}'] = {
                    'type': ltype,
                    'default': ldef,
                    'help': arg_descs[i],
                    'tuple_flag': (arg, 0)
                }
                arg_info[f'high_{arg}'] = {
                    'type': htype,
                    'default': hdef,
                    'help': arg_descs[i],
                    'tuple_flag': (arg, 1)
                }
            else:
                arg_info[arg] =  {
                    'type': arg_types[i],
                    'default': arg_defaults[i],
                    'help': arg_descs[i]
                }
        return arg_info
    
