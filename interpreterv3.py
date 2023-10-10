import copy
from enum import Enum
from intbase import InterpreterBase, ErrorType

class Tokenizer:
  # Performs tokenization and returns the tokenized program
  def tokenize_program(program):
    tokenized_program = []
    for line_num, line in enumerate(program):
      tokens = Tokenizer._tokenize(line_num, line.rstrip())
      tokenized_program.append(tokens)
    return tokenized_program

  def _remove_comment(s):
   in_quote = False
   for i in range(0,len(s)):
     if s[i] == '"':
      in_quote = not in_quote
     elif s[i] == InterpreterBase.COMMENT_DEF and not in_quote:
      return s[:i]
   return s

  def _tokenize(line_num, s):
    s = Tokenizer._remove_comment(s)

    tokens = []
    search_from = 0
    while True:
      start_quote = end_quote = None
      try:
        start_quote = s.index('"', search_from)
        end_quote = s.index('"', start_quote+1)
      except:
        if start_quote and not end_quote:
          super().error(ErrorType.SYNTAX_ERROR,f"Mismatched quotes",line_num) #no
      if start_quote is None:
        break
      else:
        tokens += s[search_from:start_quote].split()
        tokens.append(s[start_quote:end_quote+1])
        search_from = end_quote + 1
    # no more quotes found, tokenize remaining string
    tokens += s[search_from:].split()
    return tokens

class SymbolResult(Enum):
  OK = 0     # symbol created, didn't exist in top scope
  ERROR = 1  # symbol already exists in top scope

# An improved version of the EnvironmentManager that can manage a separate environment for
# each function as it executes, and has handling for nested blocks within functions
# (so variables can go out of scope once a block enters/exits).
# The internal data structure is essentially a stack (via a python list) of environments
# where each environment on the stack is a list of one or more dictionaries that map a
# variable name to a type/value. We need more than one dictionary to accomodate nested
# blocks in functions.
# If f() calls g() calls h() then while we're in function h, our stack would have
# three items on it: [[{dictionary for f}],[{dictionary for g}][{dictionary for h}]]
class EnvironmentManager:
  def __init__(self):
    self.environment = [[{}]]

  def get(self, symbol):
    nested_envs = self.environment[-1]
    for env in reversed(nested_envs):
      if symbol in env:
        return env[symbol]

    return None

  # create a new symbol in the most nested block's environment; error if
  # the symbol already exists
  def create_new_symbol(self, symbol, create_in_top_block = False):
    block_index = 0 if create_in_top_block else -1
    if symbol not in self.environment[-1][block_index]:
      self.environment[-1][block_index][symbol] = None
      return SymbolResult.OK

    return SymbolResult.ERROR

  # set works with symbols that were already created
  # it won't create a new symbol, only update it
  def set(self, symbol, value):
    nested_envs = self.environment[-1]

    for env in reversed(nested_envs):
      if symbol in env:
        env[symbol] = value
        return SymbolResult.OK

    return SymbolResult.ERROR

  # used only to populate parameters for a function call
  # and populate captured variables; use first for captured, then params
  # so params shadow captured variables
  def import_mappings(self, dict):
    cur_env = self.environment[-1][-1]
    for symbol, value in dict.items():
      cur_env[symbol] = value

  def block_nest(self):
    self.environment[-1].append({})   # [..., [{}]] -> [..., [{}, {}]]

  def block_unnest(self):
    self.environment[-1].pop()

  def push(self):
    self.environment.append([{}])       # [[...],[...]] -> [[...],[...],[]]

  def pop(self):
    self.environment.pop()

class FuncInfo:
  def __init__(self, params = None, start_ip = None):
    self.params = params  # format is [[varname1,typename1],[varname2,typename2],...]
    self.start_ip = start_ip    # line number, zero-based

class FunctionManager:
  def __init__(self, tokenized_program):
    self.func_cache = {}
    self.lambda_func = {}
    self.return_types = []  # of each line in the program
    self._cache_function_parameters_and_return_type(tokenized_program)

  # Returns a FuncInfo for the named function or lambda
  # which contains a list of params/types and the start IP of the
  # function's first instruction
  def get_function_info(self, func_name):
    if func_name not in self.func_cache:
      return None
    return self.func_cache[func_name]

  # returns true if the function name is a known function in the program
  def is_function(self, func_name):
    return func_name in self.func_cache

  # generate a synthetic function name for the lambda function, based on
  # the line number where the lambda starts
  def create_lambda_name(self,line_num):
    return InterpreterBase.LAMBDA_DEF + ':' + str(line_num)

  def lambda_captured_vars(self,funcname,environment):
    self.lambda_func[funcname] = copy.copy(environment)

  # returns the return type for the function in question
  def get_return_type_for_enclosing_function(self, line_num):
    return self.return_types[line_num]

  def _to_tuple(self, formal):
    var_type = formal.split(':')
    return (var_type[0], var_type[1])

  def _cache_function_parameters_and_return_type(self, tokenized_program):
    cur_return_type = None
    reset_after_this_line = False
    return_type_stack = [None]  # v3

    for line_num, line in enumerate(tokenized_program):
      if line and line[0] == InterpreterBase.FUNC_DEF:
        # format:  func funcname self.p1:t1 p2:t2 p3:t3 ...
        func_name = line[1]
        params = [self._to_tuple(formal) for formal in line[2:-1]]
        func_info = FuncInfo(params, line_num + 1)  # function starts executing on line after funcdef
        self.func_cache[func_name] = func_info
        return_type_stack.append(line[-1])

      if line and line[0] == InterpreterBase.LAMBDA_DEF:
        func_name = self.create_lambda_name(line_num)
        params = [self._to_tuple(formal) for formal in line[1:-1]]
        func_info = FuncInfo(params, line_num + 1)
        self.func_cache[func_name] = func_info
        return_type_stack.append(line[-1])

      if line and (line[0] == InterpreterBase.ENDFUNC_DEF or line[0] == InterpreterBase.ENDLAMBDA_DEF):
        reset_after_this_line = True

      self.return_types.append(return_type_stack[-1])  # each line in the program is assigned a return type based on
                                                 # the function it's associated with; use this to look up valid type
                                                 # for each return
      if reset_after_this_line:                  # for each line with a funcend, make sure we know the return type
        return_type_stack.pop()
        reset_after_this_line = False

# Enumerated type for our different language data types
class Type(Enum):
  INT = 1
  BOOL = 2
  STRING = 3
  VOID = 4
  FUNC = 5
  OBJECT = 6

# Represents a value, which has a type and its value
class Value:
  def __init__(self, type, value = None):
    self.t = type
    self.v = value

  def value(self):
    return self.v

  def set(self, other):
    self.t = other.t
    self.v = other.v

  def type(self):
    return self.t

# Main interpreter class
class Interpreter(InterpreterBase):
  def __init__(self, console_output=True, input=None, trace_output=False):
    super().__init__(console_output, input)
    self._setup_operations()  # setup all valid binary operations and the types they work on
    self._setup_default_values()  # setup the default values for each type (e.g., bool->False)
    self.trace_output = trace_output

  # run a program, provided in an array of strings, one string per line of source code
  def run(self, program):
    self.program = program
    self._compute_indentation(program)  # determine indentation of every line
    self.tokenized_program = Tokenizer.tokenize_program(program)
    self.func_manager = FunctionManager(self.tokenized_program)
    self.ip = self.func_manager.get_function_info(InterpreterBase.MAIN_FUNC).start_ip
    self.return_stack = []
    self.terminate = False
    self.env_manager = EnvironmentManager()   # used to track variables/scope

    # main interpreter run loop
    while not self.terminate:
      self._process_line()

  def _process_line(self):
    if self.trace_output:
      print(f"{self.ip:04}: {self.program[self.ip].rstrip()}")
    tokens = self.tokenized_program[self.ip]
    if not tokens:
      self._blank_line()
      return

    args = tokens[1:]

    match tokens[0]:
      case InterpreterBase.ASSIGN_DEF:
        self._assign(args)
      case InterpreterBase.FUNCCALL_DEF:
        self._funccall(args)
      case InterpreterBase.ENDFUNC_DEF:
        self._endfunc()
      case InterpreterBase.IF_DEF:
        self._if(args)
      case InterpreterBase.ELSE_DEF:
        self._else()
      case InterpreterBase.ENDIF_DEF:
        self._endif()
      case InterpreterBase.RETURN_DEF:
        self._return(args)
      case InterpreterBase.WHILE_DEF:
        self._while(args)
      case InterpreterBase.ENDWHILE_DEF:
        self._endwhile(args)
      case InterpreterBase.VAR_DEF: # v2 statements
        self._define_var(args)
      case InterpreterBase.LAMBDA_DEF:
        self._lambda(args)
      case InterpreterBase.ENDLAMBDA_DEF:
        self._endlambda()
      case default:
        raise Exception(f'Unknown command: {tokens[0]}')

  def _blank_line(self):
    self._advance_to_next_statement()

  def _endlambda(self):
    self._endfunc()

  def _lambda(self, tokens):
    lambda_name = self.func_manager.create_lambda_name(self.ip)
    self.func_manager.lambda_captured_vars(lambda_name,self.env_manager.environment[-1][-1])
    self._set_result(Value(Type.FUNC,lambda_name))
    for line_num in range(self.ip+1, len(self.tokenized_program)):
      tokens = self.tokenized_program[line_num]
      if not tokens:
        continue
      if tokens[0] == InterpreterBase.ENDLAMBDA_DEF and self.indents[self.ip] == self.indents[line_num]:
          self.ip = line_num + 1
          return
    super().error(ErrorType.SYNTAX_ERROR,"Missing endlambda", self.ip)

  def _assign(self, tokens):
   if len(tokens) < 2:
     super().error(ErrorType.SYNTAX_ERROR,"Invalid assignment statement")
   if '.' in tokens[0]:
     cur_mem = tokens[0].split('.')
     existing_object = self._get_value(cur_mem[0])
     if existing_object.type() != Type.OBJECT:
       super().error(ErrorType.TYPE_ERROR,f"{existing_object.value()} is not an object", self.ip)
     value_type = self._eval_expression(tokens[1:])
     existing_object.value()[cur_mem[1]] = value_type
     self._set_value(cur_mem[0],existing_object)
   else:
     value_type = self._eval_expression(tokens[1:])
     existing_value_type = self._get_value(tokens[0])
     if existing_value_type.type() != value_type.type():
       super().error(ErrorType.TYPE_ERROR,
                     f"Trying to assign a variable of {existing_value_type.type()} to a value of {value_type.type()}",
                     self.ip)
     self._set_value(tokens[0], value_type)
   self._advance_to_next_statement()

  def _funccall(self, args):
    if not args:
      super().error(ErrorType.SYNTAX_ERROR,"Missing function name to call", self.ip)
    if args[0] == InterpreterBase.PRINT_DEF:
      self._print(args[1:])
      self._advance_to_next_statement()
    elif args[0] == InterpreterBase.INPUT_DEF:
      self._input(args[1:])
      self._advance_to_next_statement()
    elif args[0] == InterpreterBase.STRTOINT_DEF:
      self._strtoint(args[1:])
      self._advance_to_next_statement()
    else:
      self.return_stack.append(self.ip+1)
      if '.' in args[0]:
        cur_mem = args[0].split('.')
        obj = self.env_manager.get(cur_mem[0])
        if obj == None:
          super().error(ErrorType.NAME_ERROR,f"{args[0]} does not exist", self.ip)
        if cur_mem[1] not in obj.value():
          super().error(ErrorType.NAME_ERROR,f"{args[0]} does not exist", self.ip)
        func = obj.value()[cur_mem[1]]
        self._create_new_environment(func.value(), args[1:])  # Create new environment, copy args into new env
        self.ip = self._find_first_instruction(func.value())
      else:
        func = self.env_manager.get(args[0])
        if func == None:
          self._create_new_environment(args[0], args[1:])  # Create new environment, copy args into new env
          self.ip = self._find_first_instruction(args[0])
        else:
          self._create_new_environment(func.value(),args[1:],InterpreterBase.LAMBDA_DEF in func.value())
          self.ip = self._find_first_instruction(func.value())

  # create a new environment for a function call
  def _create_new_environment(self, funcname, args, is_lambda = False):
    formal_params = self.func_manager.get_function_info(funcname)
    if formal_params is None:
        super().error(ErrorType.NAME_ERROR, f"Unknown function name {funcname}", self.ip)

    if len(formal_params.params) != len(args):
      super().error(ErrorType.NAME_ERROR,f"Mismatched parameter count in call to {funcname}", self.ip)

    tmp_mappings = {}
    for formal, actual in zip(formal_params.params,args):
      formal_name = formal[0]
      formal_typename = formal[1]
      arg = self._get_value(actual)
      if arg.type() != self.compatible_types[formal_typename]:
        super().error(ErrorType.TYPE_ERROR,f"Mismatched parameter type for {formal_name} in call to {funcname}", self.ip)
      if formal_typename in self.reference_types:
        tmp_mappings[formal_name] = arg
      else:
        tmp_mappings[formal_name] = copy.copy(arg)

    # create a new environment for the target function
    # and add our parameters to the env
    if is_lambda:
      for symbol, value in self.func_manager.lambda_func[funcname].items():
          if value.type() == Type.OBJECT:
            new_val = copy.copy(value.value())
            tmp_mappings[symbol] = Value(Type.OBJECT,new_val)
          else:
            tmp_mappings[symbol] = copy.copy(value)

    self.env_manager.push()
    self.env_manager.import_mappings(tmp_mappings)

  def _endfunc(self, return_val = None):
    if not self.return_stack:  # done with main!
      self.terminate = True
    else:
      self.env_manager.pop()  # get rid of environment for the function
      if return_val:
        self._set_result(return_val)
      else:
        # return default value for type if no return value is specified. Last param of True enables
        # creation of result variable even if none exists, or is of a different type
        return_type = self.func_manager.get_return_type_for_enclosing_function(self.ip)
        if return_type != InterpreterBase.VOID_DEF:
          self._set_result(self.type_to_default[return_type])
      self.ip = self.return_stack.pop()

  def _if(self, args):
    if not args:
      super().error(ErrorType.SYNTAX_ERROR,"Invalid if syntax", self.ip)
    value_type = self._eval_expression(args)
    if value_type.type() != Type.BOOL:
      super().error(ErrorType.TYPE_ERROR,"Non-boolean if expression", self.ip)
    if value_type.value():
      self._advance_to_next_statement()
      self.env_manager.block_nest()  # we're in a nested block, so create new env for it
      return
    else:
      for line_num in range(self.ip+1, len(self.tokenized_program)):
        tokens = self.tokenized_program[line_num]
        if not tokens:
          continue
        if tokens[0] == InterpreterBase.ENDIF_DEF and self.indents[self.ip] == self.indents[line_num]:
          self.ip = line_num + 1
          return
        if tokens[0] == InterpreterBase.ELSE_DEF and self.indents[self.ip] == self.indents[line_num]:
          self.ip = line_num + 1
          self.env_manager.block_nest()  # we're in a nested else block, so create new env for it
          return
    super().error(ErrorType.SYNTAX_ERROR,"Missing endif", self.ip)

  def _endif(self):
    self._advance_to_next_statement()
    self.env_manager.block_unnest()

  # we would only run this if we ran the successful if block, and fell into the else at the end of the block
  # so we need to delete the old top environment
  def _else(self):
    self.env_manager.block_unnest()   # Get rid of env for block above
    for line_num in range(self.ip+1, len(self.tokenized_program)):
      tokens = self.tokenized_program[line_num]
      if not tokens:
        continue
      if tokens[0] == InterpreterBase.ENDIF_DEF and self.indents[self.ip] == self.indents[line_num]:
          self.ip = line_num + 1
          return
    super().error(ErrorType.SYNTAX_ERROR,"Missing endif", self.ip)

  def _return(self,args):
    # do we want to support returns without values?
    return_type = self.func_manager.get_return_type_for_enclosing_function(self.ip)
    default_value_type = self.type_to_default[return_type]
    if default_value_type.type() == Type.VOID:
      if args:
        super().error(ErrorType.TYPE_ERROR,"Returning value from void function", self.ip)
      self._endfunc()  # no return
      return
    if not args:
      self._endfunc()  # return default value
      return

    #otherwise evaluate the expression and return its value
    value_type = self._eval_expression(args)
    if value_type.type() != default_value_type.type():
      super().error(ErrorType.TYPE_ERROR,"Non-matching return type", self.ip)
    self._endfunc(value_type)

  def _while(self, args):
    if not args:
      super().error(ErrorType.SYNTAX_ERROR,"Missing while expression", self.ip)
    value_type = self._eval_expression(args)
    if value_type.type() != Type.BOOL:
      super().error(ErrorType.TYPE_ERROR,"Non-boolean while expression", self.ip)
    if value_type.value() == False:
      self._exit_while()
      return

    # If true, we advance to the next statement
    self._advance_to_next_statement()
    # And create a new scope
    self.env_manager.block_nest()

  def _exit_while(self):
    while_indent = self.indents[self.ip]
    cur_line = self.ip + 1
    while cur_line < len(self.tokenized_program):
      if self.tokenized_program[cur_line][0] == InterpreterBase.ENDWHILE_DEF and self.indents[cur_line] == while_indent:
        self.ip = cur_line + 1
        return
      if self.tokenized_program[cur_line] and self.indents[cur_line] < self.indents[self.ip]:
        break # syntax error!
      cur_line += 1
    # didn't find endwhile
    super().error(ErrorType.SYNTAX_ERROR,"Missing endwhile", self.ip)

  def _endwhile(self, args):
    # first delete the scope
    self.env_manager.block_unnest()
    while_indent = self.indents[self.ip]
    cur_line = self.ip - 1
    while cur_line >= 0:
      if self.tokenized_program[cur_line][0] == InterpreterBase.WHILE_DEF and self.indents[cur_line] == while_indent:
        self.ip = cur_line
        return
      if self.tokenized_program[cur_line] and self.indents[cur_line] < self.indents[self.ip]:
        break # syntax error!
      cur_line -= 1
    # didn't find while
    super().error(ErrorType.SYNTAX_ERROR,"Missing while", self.ip)


  def _define_var(self, args):
    if len(args) < 2:
      super().error(ErrorType.SYNTAX_ERROR,"Invalid var definition syntax", self.ip)
    for var_name in args[1:]:
      if self.env_manager.create_new_symbol(var_name) != SymbolResult.OK:
        super().error(ErrorType.NAME_ERROR,f"Redefinition of variable {args[1]}", self.ip)
      # is the type a valid type?
      if args[0] not in self.type_to_default:
        super().error(ErrorType.TYPE_ERROR,f"Invalid type {args[0]}", self.ip)
      # Create the variable with a copy of the default value for the type
      if args[0] == super().OBJECT_DEF:
        self.env_manager.set(var_name, Value(Type.OBJECT,{}))
      else:
        self.env_manager.set(var_name, copy.copy(self.type_to_default[args[0]]))

    self._advance_to_next_statement()

  def _print(self, args):
    if not args:
      super().error(ErrorType.SYNTAX_ERROR,"Invalid print call syntax", self.ip)
    out = []
    for arg in args:
      val_type = self._get_value(arg)
      out.append(str(val_type.value()))
    super().output(''.join(out))

  def _input(self, args):
    if args:
      self._print(args)
    result = super().get_input()
    self._set_result(Value(Type.STRING, result))   # return always passed back in result

  def _strtoint(self, args):
    if len(args) != 1:
      super().error(ErrorType.SYNTAX_ERROR,"Invalid strtoint call syntax", self.ip)
    value_type = self._get_value(args[0])
    if value_type.type() != Type.STRING:
      super().error(ErrorType.TYPE_ERROR,"Non-string passed to strtoint", self.ip)
    self._set_result(Value(Type.INT, int(value_type.value())))   # return always passed back in result

  def _advance_to_next_statement(self):
    # for now just increment IP, but later deal with loops, returns, end of functions, etc.
    self.ip += 1

  # Set up type-related data structures
  def _setup_default_values(self):
    # set up what value to return as the default value for each type
    self.type_to_default = {}
    self.type_to_default[InterpreterBase.INT_DEF] = Value(Type.INT,0)
    self.type_to_default[InterpreterBase.STRING_DEF] = Value(Type.STRING,'')
    self.type_to_default[InterpreterBase.BOOL_DEF] = Value(Type.BOOL,False)
    self.type_to_default[InterpreterBase.VOID_DEF] = Value(Type.VOID,None)
    self.type_to_default[InterpreterBase.FUNC_DEF] = Value(Type.FUNC,FuncInfo())
    self.type_to_default[InterpreterBase.OBJECT_DEF] = Value(Type.OBJECT,{})

    # set up what types are compatible with what other types
    self.compatible_types = {}
    self.compatible_types[InterpreterBase.INT_DEF] = Type.INT
    self.compatible_types[InterpreterBase.STRING_DEF] = Type.STRING
    self.compatible_types[InterpreterBase.BOOL_DEF] = Type.BOOL
    self.compatible_types[InterpreterBase.REFINT_DEF] = Type.INT
    self.compatible_types[InterpreterBase.REFSTRING_DEF] = Type.STRING
    self.compatible_types[InterpreterBase.REFBOOL_DEF] = Type.BOOL
    self.compatible_types[InterpreterBase.FUNC_DEF] = Type.FUNC
    self.compatible_types[InterpreterBase.OBJECT_DEF] = Type.OBJECT
    self.reference_types = {InterpreterBase.REFINT_DEF, Interpreter.REFSTRING_DEF,
                            Interpreter.REFBOOL_DEF}

    # set up names of result variables: resulti, results, resultb
    self.type_to_result = {}
    self.type_to_result[Type.INT] = 'i'
    self.type_to_result[Type.STRING] = 's'
    self.type_to_result[Type.BOOL] = 'b'
    self.type_to_result[Type.FUNC] = 'f'
    self.type_to_result[Type.OBJECT] = 'o'

  # run a program, provided in an array of strings, one string per line of source code
  def _setup_operations(self):
    self.binary_op_list = ['+','-','*','/','%','==','!=', '<', '<=', '>', '>=', '&', '|']
    self.binary_ops = {}
    self.binary_ops[Type.INT] = {
     '+': lambda a,b: Value(Type.INT, a.value()+b.value()),
     '-': lambda a,b: Value(Type.INT, a.value()-b.value()),
     '*': lambda a,b: Value(Type.INT, a.value()*b.value()),
     '/': lambda a,b: Value(Type.INT, a.value()//b.value()),  # // for integer ops
     '%': lambda a,b: Value(Type.INT, a.value()%b.value()),
     '==': lambda a,b: Value(Type.BOOL, a.value()==b.value()),
     '!=': lambda a,b: Value(Type.BOOL, a.value()!=b.value()),
     '>': lambda a,b: Value(Type.BOOL, a.value()>b.value()),
     '<': lambda a,b: Value(Type.BOOL, a.value()<b.value()),
     '>=': lambda a,b: Value(Type.BOOL, a.value()>=b.value()),
     '<=': lambda a,b: Value(Type.BOOL, a.value()<=b.value()),
    }
    self.binary_ops[Type.STRING] = {
     '+': lambda a,b: Value(Type.STRING, a.value()+b.value()),
     '==': lambda a,b: Value(Type.BOOL, a.value()==b.value()),
     '!=': lambda a,b: Value(Type.BOOL, a.value()!=b.value()),
     '>': lambda a,b: Value(Type.BOOL, a.value()>b.value()),
     '<': lambda a,b: Value(Type.BOOL, a.value()<b.value()),
     '>=': lambda a,b: Value(Type.BOOL, a.value()>=b.value()),
     '<=': lambda a,b: Value(Type.BOOL, a.value()<=b.value()),
    }
    self.binary_ops[Type.BOOL] = {
     '&': lambda a,b: Value(Type.BOOL, a.value() and b.value()),
     '==': lambda a,b: Value(Type.BOOL, a.value()==b.value()),
     '!=': lambda a,b: Value(Type.BOOL, a.value()!=b.value()),
     '|': lambda a,b: Value(Type.BOOL, a.value() or b.value())
    }

  def _compute_indentation(self, program):
    self.indents = [len(line) - len(line.lstrip(' ')) for line in program]

  def _find_first_instruction(self, funcname):
    func_info = self.func_manager.get_function_info(funcname)
    if not func_info:
      super().error(ErrorType.NAME_ERROR,f"Unable to locate {funcname} function")

    return func_info.start_ip

  # given a token name (e.g., x, 17, True, "foo"), give us a Value object associated with it
  def _get_value(self, token):
    if not token:
      super().error(ErrorType.NAME_ERROR,f"Empty token", self.ip)
    if token[0] == '"':
      return Value(Type.STRING, token.strip('"'))
    if token.isdigit() or token[0] == '-':
      return Value(Type.INT, int(token))
    if token == InterpreterBase.TRUE_DEF or token == Interpreter.FALSE_DEF:
      return Value(Type.BOOL, token == InterpreterBase.TRUE_DEF)

    # look in environments for variable
    if '.' in token:
      cur_mem = token.split('.')
      cur_obj = self.env_manager.get(cur_mem[0])
      if cur_mem[1] not in cur_obj.value():
        super().error(ErrorType.NAME_ERROR,f"Unknown variable {cur_mem[1]} in {cur_mem[0]}", self.ip)
      val = cur_obj.value()[cur_mem[1]]
    else:
      val = self.env_manager.get(token)
    if val != None:
      return val
    
    if self.func_manager.is_function(token):
      val = Value(Type.FUNC,token)
      return val
    
    # not found
    super().error(ErrorType.NAME_ERROR,f"Unknown variable {token}", self.ip)

  # given a variable name and a Value object, associate the name with the value
  def _set_value(self, varname, to_value_type):
    value_type = self.env_manager.get(varname)
    if value_type == None:
      super().error(ErrorType.NAME_ERROR,f"Assignment of unknown variable {varname}", self.ip)
    value_type.set(to_value_type)

  # bind the result[s,i,b] variable in the calling function's scope to the proper Value object
  def _set_result(self, value_type):
    # always stores result in the highest-level block scope for a function, so nested if/while blocks
    # don't each have their own version of result
    result_var = InterpreterBase.RESULT_DEF + self.type_to_result[value_type.type()]
    self.env_manager.create_new_symbol(result_var, True)  # create in top block if it doesn't exist
    if value_type.type() == Type.OBJECT:
      new_dict = {}
      for symbol,item in value_type.value().items():
        new_dict[symbol] = copy.copy(item)
      self.env_manager.set(result_var,Value(Type.OBJECT,new_dict))
    else:
      self.env_manager.set(result_var, copy.copy(value_type))

  # evaluate expressions in prefix notation: + 5 * 6 x
  def _eval_expression(self, tokens):
    stack = []

    for token in reversed(tokens):
      if token in self.binary_op_list:
        v1 = stack.pop()
        v2 = stack.pop()
        if v1.type() != v2.type():
          super().error(ErrorType.TYPE_ERROR,f"Mismatching types {v1.type()} and {v2.type()}", self.ip)
        operations = self.binary_ops[v1.type()]
        if token not in operations:
          super().error(ErrorType.TYPE_ERROR,f"Operator {token} is not compatible with {v1.type()}", self.ip)
        stack.append(operations[token](v1,v2))
      elif token == '!':
        v1 = stack.pop()
        if v1.type() != Type.BOOL:
          super().error(ErrorType.TYPE_ERROR,f"Expecting boolean for ! {v1.type()}", self.ip)
        stack.append(Value(Type.BOOL, not v1.value()))
      elif '.' in token:
        cur_mem = token.split('.')
        cur_obj = self._get_value(cur_mem[0])
        if cur_obj.type() != Type.OBJECT:
          super().error(ErrorType.TYPE_ERROR,f"{cur_obj.value()} is not an object", self.ip)
        if cur_mem[1] not in cur_obj.value():
          super().error(ErrorType.NAME_ERROR,f"Unknown variable {cur_mem[1]} in {cur_mem[0]}", self.ip)
        value_type = cur_obj.value()[cur_mem[1]]
        stack.append(value_type)
      else:
        value_type = self._get_value(token)
        stack.append(value_type)

    if len(stack) != 1:
      super().error(ErrorType.SYNTAX_ERROR,f"Invalid expression", self.ip)

    return stack[0]