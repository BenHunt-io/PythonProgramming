# The with statement and context managers
# --- simplifies resource management patterns ---


with open('hello.txt', 'w') as fout:
		fout.write('hello')




# What it equates to

fout = open('hello.txt', 'w')
try:
	fout.write('hello')
finally:
	fout.close()


# Context Managers --- just need to implement enter/exit in your own class.
class ManagedFile:
	def __init__(self,name):
		self.name = name

	def __enter__(self):
		self.file = open(self.name, 'w')
		return self.file

	def __exit__(self, exc_type, exc_val, exc_tb):
		if self.file:
			self.file.close()

with ManagedFile('hello.txt') as f:  # assigns what is returned from __enter__ to f
	f.write('hello')