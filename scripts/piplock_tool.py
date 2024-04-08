import sys

file = sys.argv[1]
with open(file) as f1:
    text = f1.read()
package_list = text.split("Would install ")[1].split(" ")
content = "\n".join(["==".join(i.rsplit("-", 1)) for i in package_list])
with open(file, "w") as f2:
    f2.write(content)
