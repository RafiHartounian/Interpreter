# Custom Interpreter 

## Overview

This is a custom interpreter written in Python (interpreterv3.py) that executes programs provided in an array of strings. The interpreter supports various language constructs, including variable assignments, function calls, conditional statements, loops, and more.

## Features

- **Tokenization:** The interpreter performs tokenization on the input program to break it down into meaningful units.

- **Environment Management:** The interpreter manages environments for variable scope, allowing variables to go out of scope within nested blocks.

- **Function Handling:** The interpreter supports function definitions, lambda functions, and function calls. It manages function parameters, return types, and handles nested functions.

- **Type System:** The interpreter has a type system with support for integers, booleans, strings, functions, and objects. It performs type checking during assignments and operations.

- **Control Flow:** The interpreter supports conditional statements (`if`, `else`, `endif`), loops (`while`, `endwhile`), and function returns.

## Usage

To use the interpreter, create an instance of the `Interpreter` class and call the `run` method with the program provided as an array of strings:

```python
from interpreterv3 import Interpreter

program = [
    "var x:int",
    "x = 42",
    "print(x)",
]

interpreter = Interpreter()
interpreter.run(program)
```

## Language Constructs
### Variable Definition
```python
var variable_name:type
```
### Assignment
```python
variable_name = expression
```
### Function Definition
```python
func function_name param1:type param2:type ... return_type
    # Function body
endfunc
```
### Lambda Function
```python
lambda param1:type param2:type ... return_type
    # Lambda body
endlambda
```
### Function Call
```python
func_name argument1 argument2 ...
```
### Conditional Statements
```python
if condition
    # Code block for true condition
else
    # Code block for false condition
endif
```
### Loops
```python
while condition
    # Loop body
endwhile
```
### Return Statement
```python
return expression
```
### Print Statement
```python
print(expression)
```
### Input Statement
```python
input("Prompt: ")
```
### Type Conversion (String to Int)
```python
strtoint(string_expression)
```

## Notes
The interpreter uses a prefix notation for expressions.

The language supports basic arithmetic operations (+, -, *, /, %), comparison operations (==, !=, <, <=, >, >=), logical operations (&, |), and the negation operator (!).

Objects in the language are represented as dictionaries.

The interpreter provides basic error handling for syntax errors, type errors, and name errors.

Feel free to explore and modify the interpreter for your specific needs!
