'''
 Library for progress bar display on console
 Created - 19.9.15
'''
from __future__ import print_function
import sys

def progress_bar(percent_complete, show_percentage=False, align_right=False, bar_body = ".", bar_empty = " ", bar_begin = "[", bar_end = "]", bar_size=20, bar_arrow=None, initial_text = '', ending_text = '\n'):
	''' Function returns 1 on completion

		Parameters
		----------
		- percent_complete   integer value
		- show_percentage    boolean to show percentage value after the bar
		- align_right        align percentage bar to the right
		- bar_body           character element of the filled bar
		- bar_empty          character element of the empty bar
		- bar_begin          left delimiter of the bar
		- bar_end            right delimiter of the bar
		- bar_size           size of the bar in characters
		- bar_arrow          character on the tip of the filled bar
		- initial_text       text to show before the bar
		- ending_text        text to show after the bar once loading finishes

		Returns
		-------
		Boolean True if completed, False otherwise

		Example
		-------
		>>> for x in range(101):
			progress_bar(x, show_percentage=True, bar_body="-", bar_empty=" ", bar_size=20, bar_arrow=">", initial_text="Testing: ", ending_text=" Finished!")
	'''

	# Clamp percenteage between 0-100
	percent_complete = 0 if percent_complete < 0 else (100 if percent_complete > 100 else percent_complete)


	# Bar has a body with a maximum of bar_size spaces
	dots_to_print = int(bar_size/100.0 * percent_complete)
	empty_spaces = int(bar_size - dots_to_print) - 1 if bar_arrow != "" else int(bar_size - dots_to_print)
	bar = bar_begin + bar_body * dots_to_print + (bar_arrow if bar_arrow and percent_complete != 100 else "") + bar_empty * empty_spaces + bar_end
	# spacing to align right
	if align_right:
		spacing = '\033[?7l'+' '*400+'\033[?7h\033['+ str(bar_size + (9 if show_percentage else 1))+'D'

# PRINT VERSION
	if not show_percentage: print("\r" + initial_text + (spacing if align_right else '') + bar, end='')
	else: print("\r" + initial_text + (spacing if align_right else '') + bar + "% 5d %%  " % percent_complete, end='')
	sys.stdout.flush()
	if percent_complete == 100:
		print(ending_text, end='')
		sys.stdout.flush()
		return True # return 1 if complete
	return False

# NO PRINT VERSION
	#if not show_percentage: return bar
	#else: return bar + "  %.0f %%" % percent_complete

if __name__ == "__main__":
	import time
	print("Showing example")
	for x in range(101):
		progress_bar(x, show_percentage=True, bar_body="-", bar_empty=" ", bar_size=20, bar_arrow=">", initial_text = "Testing: ", ending_text = " Finished!\n")
		time.sleep(0.04)

