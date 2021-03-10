
style_string = """
<style>

.container { width:100% !important; }

.hit {
        border-style: dotted;
        border-width: 20px;
        border-color: #ddd;
        color: black;
        background-color: white;
        }
.miss {
        border-style: solid;
        border-width: 20px;
        border-color: black;
        color: pink;
        background-color: black;
        }
.outside {
        border-style: solid;
        border-width: 10px;
        border-color: #ddf;
        }
</style>
"""

from IPython.display import display, HTML

table_template = """
<div class="outside">
<table class="outside">
    <tr>
    %s
    </tr>
</table>
</div>
"""

hit_template = """
<td><div class="hit"> %s </div></td>
"""

miss_template = """
<td><div class="miss"> %s </div></td>
"""

def classification_table(zero_one_string):
    L = []
    for c in zero_one_string:
        if int(c):
            entry = hit_template % c
        else:
            entry = miss_template % c
        L.append(entry)
    body = "".join(L)
    return table_template % body

def cls(zero_one_string):
    tb = classification_table(zero_one_string)
    display(HTML(tb))

def go():
    display(HTML(style_string))

def compare_metric_table():
    display(HTML(comparison_table))

comparison_table = """

<table border>
    <tr>
        <th colspan="8">Secondary Statistic</th>
    </tr>
    <tr>
        <td>
            <em>Primary Statistic Favors</em>
        </td>
        <th>
            ASP = Average Squared Preference
        </th>
        <th>
            AUROC = Area Under the ROC Curve
        </th>
        <th>
            AUPR = Area Under the precision recall curve
        </th>
        <th>
            ALP = Average Log Preference
        </th>
        <th>
            RLP = Reversed Log Preference
        </th>
        <th>
            SFP = Squared False Penalty
        </th>
        <th>
            LFP = Log False Penalty
        </th>
    </tr>
    
    <tr>
        <th>
            ASP = Average Squared Preference
        </th>
        
        
        <td>
            <!-- ASP = Average Squared Preference -->
            (same)
        </td>
        <td>
            <!-- AUROC = Area Under the ROC Curve -->
            hits spread out
        </td>
        <td>
            <!-- AUPR = Area Under the precision recall curve -->
            ???
        </td>
        <td>
            <!-- ALP = Average Log Preference -->
            hits spread out
        </td>
        <td>
            <!-- RLP = Reversed Log Preference -->
            hits clustered
        </td>
        <td>
            <!-- SFP = Squared False Penalty -->
            hits spread out
        </td>
        <td>
            <!-- LFP = Log False Penalty -->
            hits clustered
        </td>
    </tr>
    <tr>
        <th>
            AUROC = Area Under the ROC Curve
        </th>
        
        <td>
            <!-- ASP = Average Squared Preference -->
            hits clustered
        </td>
        <td>
            <!-- AUROC = Area Under the ROC Curve -->
            (same)
        </td>
        <td>
            <!-- AUPR = Area Under the precision recall curve -->
            tolerates early misses
        </td>
        <td>
            <!-- ALP = Average Log Preference -->
            hits spread out
        </td>
        <td>
            <!-- RLP = Reversed Log Preference -->
            hits clustered
        </td>
        <td>
            <!-- SFP = Squared False Penalty -->
            hits spread out
        </td>
        <td>
            <!-- LFP = Log False Penalty -->
            hits clustered
        </td>
    </tr>
    <tr>
        <th>
            AUPR = Area Under the precision recall curve
        </th>
        <td>
            <!-- ASP = Average Squared Preference -->
            favors early hits
        </td>
        <td>
            <!-- AUROC = Area Under the ROC Curve -->
            favors early hits
        </td>
        <td>
            <!-- AUPR = Area Under the precision recall curve -->
            (same)
        </td>
        <td>
            <!-- ALP = Average Log Preference -->
            favors early hits
        </td>
        <td>
            <!-- RLP = Reversed Log Preference -->
            favors early hits
        </td>
        <td>
            <!-- SFP = Squared False Penalty -->
            favors early hits
        </td>
        <td>
            <!-- LFP = Log False Penalty -->
            favors early hits
        </td>
    </tr>
    <tr>
        <th>
            ALP = Average Log Preference
        </th>
        <td>
            <!-- ASP = Average Squared Preference -->
            hits clustered
        </td>
        <td>
            <!-- AUROC = Area Under the ROC Curve -->
            hits clustered
        </td>
        <td>
            <!-- AUPR = Area Under the precision recall curve -->
            tolerates early misses
        </td>
        <td>
            <!-- ALP = Average Log Preference -->
            (same)
        </td>
        <td>
            <!-- RLP = Reversed Log Preference -->
            hits clustered
        </td>
        <td>
            <!-- SFP = Squared False Penalty -->
            hits clustered
        </td>
        <td>
            <!-- LFP = Log False Penalty -->
            hits clustered
        </td>
    </tr>
        <th>
            RLP = Reversed Log Preference
        </th>
        <td>
            <!-- ASP = Average Squared Preference -->
            hits spread out
        </td>
        <td>
            <!-- AUROC = Area Under the ROC Curve -->
            hits spread out
        </td>
        <td>
            <!-- AUPR = Area Under the precision recall curve -->
            tolerates early misses
        </td>
        <td>
            <!-- ALP = Average Log Preference -->
            hits spread out
        </td>
        <td>
            <!-- RLP = Reversed Log Preference -->
            (same)
        </td>
        <td>
            <!-- SFP = Squared False Penalty -->
            hits spread out
        </td>
        <td>
            <!-- LFP = Log False Penalty -->
            CORRELATED!
        </td>
    </tr>
    <tr>
        <th>
            SFP = Squared False Penalty
        </th>
        
        <td>
            <!-- ASP = Average Squared Preference -->
            hits clustered
        </td>
        <td>
            <!-- AUROC = Area Under the ROC Curve -->
            hits clustered
        </td>
        <td>
            <!-- AUPR = Area Under the precision recall curve -->
            tolerates early misses
        </td>
        <td>
            <!-- ALP = Average Log Preference -->
            hits clustered
        </td>
        <td>
            <!-- RLP = Reversed Log Preference -->
            hits clustered
        </td>
        <td>
            <!-- SFP = Squared False Penalty -->
            (same)
        </td>
        <td>
            <!-- LFP = Log False Penalty -->
            hits clustered
        </td>
    </tr>
</table>
"""

go()
