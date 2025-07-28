# module to send commands from Python to VND via ZMQ
# 
# Example call:
# mySel=NeuronSel('soma x < -30 && population == internal')
# mySel.get(['node_id','population'])
# VND.add_rep(sel=mySel, color='red', style='morpho_line', material ='Opaque', show='True')  (
#   except initial sel options optional
import re
import ast
import shlex
import sys
import time
import zmq
context = zmq.Context()
#sender = context.socket(zmq.PUSH)
#sender.connect("tcp://localhost:5556")
#receiver = context.socket(zmq.PULL)
#receiver.connect("tcp://localhost:5557")
#receiver.RCVTIMEO=5000
requester= context.socket(zmq.REQ)
requester.connect("tcp://localhost:5554")
requester.RCVTIMEO=5000



def restart_vnd_zmq():
    context = zmq.Context()
    #sender = context.socket(zmq.PUSH)
    #sender.connect("tcp://localhost:5556")
    #receiver = context.socket(zmq.PULL)
    #receiver.connect("tcp://localhost:5557")
    #receiver.RCVTIMEO=5000
    requester= context.socket(zmq.REQ)
    requester.connect("tcp://localhost:5554")
    requester.RCVTIMEO=5000
    
def vndSocketReq(s):
    requester.send(s.encode())
    r = requester.recv()
    return r.decode()

def tcl_nested_list_to_py(tcl_str):
    """
    Converts a Tcl-style list string into a Python list.
    Handles both flat lists (e.g., '1 2 3') and simple nested lists (e.g., '{1 2} {3 4}').
    """
    tcl_str = tcl_str.strip()

    # Check for nested list pattern: one or more {...} groups
    if re.search(r'\{[^{}]*\}', tcl_str):
        # Nested list: extract each {...} and split contents
        sublists = re.findall(r'\{([^{}]*)\}', tcl_str)
        return [sub.strip().split() for sub in sublists]
    else:
        # Flat list: just split on whitespace
        return tcl_str.split()
    

def tcl_simple_list_to_py(tcl_str):
    return tcl_str.strip().split()

def tcl_list_to_py(tcl_str):
    """
    Convert a Tcl-style list string to a Python list.
    Handles flat, nested, and mixed lists using recursive parsing.
    """
    def parse(tokens):
        result = []
        while tokens:
            token = tokens.pop(0)
            if token.startswith('{') and token.endswith('}'):
                # Strip outer braces and re-tokenize the content
                inner = token[1:-1]
                inner_tokens = shlex.split(inner, posix=True)
                result.append(parse(inner_tokens))
            else:
                result.append(token)
        return result

    # Use shlex to tokenize, respecting quoted/braced strings
    tokens = shlex.split(tcl_str, posix=True)
    return parse(tokens)

def bytes_to_str(data, encoding='utf-8', errors='strict'):
    """
    Converts a bytes-like object to a string.
    
    Parameters:
        data (bytes or bytearray): The data to convert.
        encoding (str): Encoding to use (default: 'utf-8').
        errors (str): Error handling ('strict', 'ignore', 'replace').

    Returns:
        str: The decoded string.
    """
    if isinstance(data, (bytes, bytearray)):
        return data.decode(encoding, errors)
    elif isinstance(data, str):
        return data  # Already a string
    else:
        raise TypeError(f"Expected bytes-like object or str, got {type(data)}")
        
def py_list_to_tcl(py_list):
    """
    Converts a Python list (flat or nested) into a Tcl-style list string.

    Example:
        [[1, 2, 3], [4, 5, 6]] → "{1 2 3} {4 5 6}"
        [1, 2, 3]             → "1 2 3"

    Returns:
        str: Tcl-style list string.
    """
    if not isinstance(py_list, list):
        raise TypeError("Input must be a list")

    def format_sublist(sub):
        if isinstance(sub, list):
            return "{" + " ".join(str(item) for item in sub) + "}"
        else:
            return str(sub)

    if all(isinstance(el, list) for el in py_list):
        return " ".join(format_sublist(sub) for sub in py_list)
    else:
        return " ".join(str(item) for item in py_list)  
    
    
class NeuronSel:
    def __init__(self, sel_string):
        gid_list = vndSocketReq(f'::neuro::parse_full_selection_string "{sel_string}" node')
        if gid_list == "VND_ERROR":
            raise ValueError (f'Failed to create NeuronSel: vndSocketReq returned VND_ERROR')
        self.sel_text = sel_string                      
        self.items=tcl_simple_list_to_py(gid_list)

    def __repr__(self):
        return f"NeuronSel(len={len(self.items)})" #just give number of items, don't show user gid list  

    def get(self, attrib_list):
        tcl_attrib_list = py_list_to_tcl(attrib_list)
        tcl_gid_list = py_list_to_tcl(self.items)
        return tcl_nested_list_to_py( vndSocketReq(f'::neuro::query_node_list_attrib_values "{tcl_attrib_list}" "{tcl_gid_list}"'))

    def get_events_dict(self,population_name):
        # convert dict from function format to a literal format for safe evaluation
        funcdict= vndSocketReq(f'::NeuronVND:spike_emit "{self.sel_text}" {population_name}')
        #extract and rename lists from JavaScript-style dict to Python
        # Match key = [ ... ] entries
        pattern = r'(\w+)\s*=\s*(\[[^\]]*\])'
        matches = re.findall(pattern, funcdict)

        # Mapping of original keys to desired output keys
        key_map = {'x': 't', 'y': 'n', 'c': 'c', 'g': 'g'}
        result = {}
        
        for key, list_str in matches:
            if key in key_map:
                result[key_map[key]] = ast.literal_eval(list_str)

        return result

    def get_events_plot_dict(self,population_name):
      # convert dict from funtion format to literal format for safe evaluation
      funcdict= vndSocketReq(f'::NeuronVND:spike_emit "{self.sel_text}" {population_name}')
      # Match key = [ ... ] entries
      pattern = r'(\w+)\s*=\s*(\[[^\]]*\])'
      matches = re.findall(pattern, funcdict)

      # Extract only the desired keys
      desired_keys = {'x', 'y', 'c', 'g'}
      result = {}
      for key, list_str in matches:
          if key in desired_keys:
              result[key] = ast.literal_eval(list_str)

      return result

    
def add_rep ( *, sel:NeuronSel, color="Type", style = "soma", material="Opaque", show = "True", scaling="1.0", resolution="6"):
    #ss = f'::NeuronVND::createRepArgs style {style} color {color} material {material} show {show} selection "gid == {py_list_to_tcl(sel.get(["gid"]))}"'
    ss = f'::NeuronVND::createRepArgs style {style} color {color} material {material} show {show} scaling {scaling} resolution {resolution} selection "{sel.sel_text}"'
    r = vndSocketReq(ss) 
    return r
                                                                                                                     
                                                                                                                     
                                                                                                                     
def del_rep(repid):
    ss = f'::NeuronVND::delRepByRepid {repid}'
    r = vndSocketReq(ss) 
    return r
    
def list_reps():
    ss = f'::neuro::cmd_query rep_list'
    r = vndSocketReq(ss) 
    return tcl_simple_list_to_py(r)

def list_rep_properties():
    ss = f'::neuro::cmd_query rep_property_list'
    r = vndSocketReq(ss) 
    return tcl_list_to_py(r)

def list_attributes():
    ss = f'lindex [list "[::neuro::cmd_query standard_node_attribs] [::neuro::cmd_query non_standard_node_attribs]"] 0'
    r = vndSocketReq(ss) 
    return tcl_simple_list_to_py(r)
    
def mod_rep():
    return 

#def array_string(a):
#  ss = ""
#  for x in a:
#      ss = " ".join([ss,str(x)])
#  return ss
#
# Old routines:
# VND.create_rep(node_ids= exc_cell_ids, population = 'internal', color='cyan', material = 'Opaque', style="soma", show='True',comm_sender=sender)
#def create_rep(node_ids,population,color,style,material,show,comm_sender):
#  num_neur = len(node_ids)
##  print('num_neurons=', num_neur) 
#  print('population=', population)
#  print ('color=', color)
#  print ('style=', style)
#  print ('show=', show)
#  ss = " ".join(['::NeuronVND::createRepArgs  style',style, 'color', color, 'material', material, 'show', show, 'num_neurons', str(num_neur), 'selection "(population ==', population, ') && (node ==', array_string(node_ids), ')"'])  
#  #ss = " ".join(['::neuro::cmd_create_rep_node_fullsel ',style, color, material,str(1.0), str(6), '"(population ==', population, ') && (node ==', array_string(node_ids), ')"'])  
  #print ('ss=', ss)
#  comm_sender.send(ss.encode('utf8'))
 
