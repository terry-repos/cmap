from pyutils.command_history import CommandHistory
import inspect

if 'g' in globals():
	del globals()['g']

command_history = CommandHistory()
dat = {}