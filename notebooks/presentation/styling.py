
style_string = """
<style>

.container { width:100% !important; }

.hit {
        border-style: dotted;
        border-width: 20px;
        border-color: #ddd;
        }
.miss {
        border-style: solid;
        border-width: 20px;
        border-color: black;
        }
.outside {
        border-style: solid;
        border-width: 10px;
        border-color: #ddf;
        }
</style>
"""

from IPython.display import display, HTML

def go():
    display(HTML(style_string))

go()
