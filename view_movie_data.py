import pandas
import webbrowser
import os

data_table = pandas.read_csv("test.txt", sep=",", encoding="ISO-8859-1")

html = data_table[0:200].to_html()

with open("data.html", "w") as f:
    f.write(html)

path = os.path.abspath("data.html")
webbrowser.open("file://{}".format(path))