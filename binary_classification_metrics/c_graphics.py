
from jp_doodle import dual_canvas
from jp_doodle import array_image
import jp_proxy_widget
import ipywidgets as widgets
import numpy as np
from . import c_stats as st
from . import combinatorics_helpers as ch
from .import ranked_combinations as rc
from IPython.display import display

class RankOfRankingsViz:

    def __init__(
        self, 
        primary_stat=st.AUPR, 
        secondary_stat=st.AUROC, 
        n=7, 
        k=2, 
        max_n=100, 
        width=800,
        combinations_array=None,
        ):
        (self.primary_stat, self.secondary_stat, self.n, self.k, self.max_n) = (primary_stat, secondary_stat, n, k, max_n)
        self.width = width
        self.rank_of_rankings = None
        self.selected_rank = None
        self.scatter = None
        self.curves = None
        self.combinations_array = None
        self.check_change_array = widgets.Checkbox(
            value=(combinations_array is None),
            description='new array',
            disabled=False,
        )
        self.recalculate(combinations_array)
        self.make_widget()

    def show(self):
        display(self.assembly)

    def recalculate(self, combinations_array=None):
        if combinations_array is not None:
            self.combinations_array = combinations_array
        elif (self.combinations_array is None) or (self.check_change_array.value):
            self.combinations_array = ch.limited_combinations(self.n, self.k)
        self.ranker = rc.Ranker(self.combinations_array)
        self.rank = self.ranker.rank(self.primary_stat, self.secondary_stat)
        self.recalculate_rank()

    def recalculate_rank(self, index=0):
        self.combo = self.rank.combination(index)
        self.curve_infos = [self.combo.curve_info(i) for i in (1, 2)]

    def make_widget(self):
        width = self.width
        self.info = widgets.HTML("<div>Ranker of rankings explorer.</div>")
        self.n_k_slider = n_k_slider(self.set_n_k)
        stat_abbrevs = sorted(st.ABBREVIATION_TO_STATISTIC.keys())
        self.primary_dropdown = widgets.Dropdown(
            options=stat_abbrevs, value=self.primary_stat.abbreviation, description='primary stat')
        self.secondary_dropdown = widgets.Dropdown(
            options=stat_abbrevs, value=self.secondary_stat.abbreviation, description='secondary stat')
        self.reset_button = widgets.Button(description="reset stats")
        self.reset_button.on_click(self.reset_stats)
        self.stats_box = widgets.HBox([
            self.primary_dropdown, self.secondary_dropdown, self.reset_button, self.check_change_array,
        ])
        hwidth = width * 0.5
        self.scatter = dual_canvas.DualCanvasWidget(width=hwidth, height=hwidth)
        self.curves = dual_canvas.DualCanvasWidget(width=hwidth, height=hwidth)
        self.plots_box = widgets.HBox([
            self.scatter, 
            self.curves,
        ])
        self.rebuild()
        self.assembly = widgets.VBox([
            self.info,
            self.n_k_slider,
            self.stats_box,
            self.rank_of_rankings,
            self.plots_box,
            self.selected_rank,
        ])
        return self.assembly

    def reset_stats(self, *ignored_args):
        p_abbrev = self.primary_dropdown.value
        s_abbrev = self.secondary_dropdown.value
        self.primary_stat = st.ABBREVIATION_TO_STATISTIC[p_abbrev]
        self.secondary_stat = st.ABBREVIATION_TO_STATISTIC[s_abbrev]
        self.recalculate()
        self.rebuild()

    def rebuild(self):
        with self.scatter.delay_redraw():
            self.draw_scatter()
        with self.curves.delay_redraw():
            self.draw_curves()
        self.rebuild_arrays()

    colors = "green magenta".split()

    def draw_curves(self):
        hwidth = 0.5 * self.width
        curves = self.curves
        curves.reset_canvas()
        # = curves.frame_region(0, 0, hwidth, hwidth, 0, 0, 1.0, 1.0)
        #background = background_frame.frame_rect(0, 0, 1.0, 1.0, color="cornsilk", name=True)
        frames = []
        colors = self.colors
        widths = [6,3]
        c = self.combo
        cis = self.curve_infos
        # share y axis
        all_points = np.array(list(cis[0].points) + list(cis[1].points))
        (_xmax, ymax) = all_points.max(axis=0)
        for index in (0,1):
            ci = self.curve_infos[index]
            color = colors[index]
            width = widths[index]
            points = ci.points
            apoints = np.array(points)
            (xmax, _ymax) = apoints.max(axis=0)
            frame = curves.frame_region(0, 0, hwidth, hwidth, 0, 0, xmax, ymax)
            frames.append(frame)
            marker = frame.circle(x=0, y=0, color="red", r=9, name=True, events=False)
            lastpoint = None
            for (threshold, p) in enumerate(points):
                (x, y) = p
                circ = frame.circle(x=x, y=y, color=color, r=7, name=True)
                curve_circle_callbacks(circ, frame, threshold, self, x, y, marker)
                if lastpoint is not None:
                    (x1, y1) = lastpoint
                    frame.line(x1, y1, x, y, color=color, lineWidth=width)
                lastpoint = p
            if index==0:
                frame.lower_left_axes(
                    0, 0, xmax, ymax,
                    tick_line_config= {"color": color},
                    tick_text_config= {"color": color},
                    )
            else:
                """frame.right_axis(
                    min_value= 0,
                    max_value= ymax,
                    #max_tick_count= 6,
                    axis_origin= {"x": xmax, "y": 0},
                    tick_line_config= {"color": color},
                    tick_text_config= {"color": color},
                    )"""
                frame.top_axis(
                    min_value= 0,
                    max_value= xmax,
                    #max_tick_count= 4,
                    axis_origin= {"x": 0, "y": ymax},
                    tick_line_config= {"color": color},
                    tick_text_config= {"color": color},
                    #add_end_points= True,
                    )
        curves.text(hwidth * 0.1, hwidth * 0.1, c.metric1.abbreviation + "=" + str(c.r1)[:5], color=colors[0], background="white")
        curves.text(hwidth * 0.1, hwidth * 0.2, c.metric2.abbreviation + "=" + str(c.r2)[:5], color=colors[1], background="white")
        curves.fit(margin=10)
        self.scatter_highlight.change(x=c.r1, y=c.r2)

    def draw_scatter(self):
        hwidth = 0.5 * self.width
        scatter = self.scatter
        scatter.reset_canvas()
        points = self.rank.pairs
        [xmax, ymax] = points.max(axis=0)
        [xmin, ymin] = points.min(axis=0)
        xlimit = max(xmax, 1)
        ylimit = max(ymax, 1)
        frame = scatter.frame_region(0, 0, hwidth, hwidth, 0, 0, xlimit, ylimit)
        background = frame.frame_rect(0, 0, 1, 1, name=True, color="#eef")
        frame.frame_rect(xmin, ymin, xmax-xmin, ymax-ymin, color="cornsilk")
        background.on("mousemove", self.scatter_move)
        background.on("click", self.scatter_click)
        frame.lower_left_axes(0, 0, xmax, ymax)
        for (x, y) in points:
            frame.circle(x=x, y=y, color="#555", r=2)
        c = self.combo
        self.scatter_highlight = frame.circle(x=c.r1, y=c.r2, r=5, name=True, events=False, color="red")
        qwidth = hwidth * 0.5
        cs = self.colors
        def label_txt(abbrev, mn, Mx):
            return "%s: %4.2f : %4.2f" % (abbrev, mn, Mx)
        xlabel = label_txt(c.metric1.abbreviation, xmin, xmax)
        scatter.text(x=qwidth, y=hwidth, text=xlabel, align="center", background="white", color=cs[0])
        ylabel = label_txt(c.metric2.abbreviation, ymin, ymax)
        scatter.text(
            x=hwidth, y=qwidth, text=ylabel, align="center", background="white", degrees=90, color=cs[1])
        scatter.fit(margin=10)

    def scatter_click(self, event):
        index = self.scatter_nearest(event)
        self.set_rank_index(index)

    def scatter_move(self, event):
        return self.scatter_nearest(event)

    def scatter_nearest(self, event):
        position = event['model_location']
        x = float(position["x"])
        y = float(position["y"])
        (index, nearest) = self.rank.nearest_pair([x, y])
        (nx, ny) = nearest
        self.scatter_highlight.change(x=nx, y=ny)
        return index

    def rebuild_arrays(self):
        width = self.width
        sorted_array = self.rank.array
        self.rank_of_rankings = array_image.show_array(
            sorted_array, width=width, height=200, background="cyan", hover_text_callback=self.rankings_hover,
            widget=self.rank_of_rankings, margin=20,
        )
        self.rank_of_rankings.image_display.on("click", self.rankings_click)
        self.rebuild_combo_array()

    def rankings_click(self, event):
        position = event['model_location']
        x = int(position["x"])
        #y = int(position["y"])
        self.set_rank_index(x)

    def set_rank_index(self, x):
        self.recalculate_rank(x)
        self.draw_curves()
        self.rebuild_combo_array()

    def rebuild_combo_array(self, threshold=None):
        width = self.width
        combo = self.combo.combo_array
        cstr = combo_str(combo) #"".join([str(c) for c in combo])
        self.rank_of_rankings.hover_info.change(text=cstr)
        self.info.value = "Selected: " + cstr
        ln = len(combo)
        if threshold is None:
            threshold = ln
        combo_array = np.zeros((1, len(combo), 3), dtype=np.int)
        for i in range(ln):
            combo_array[0,i] = combo[i] * 10
            if i > threshold:
                #combo_array[0, i, 0] = 1 - combo[i]
                combo_array[0,i] = 1 + combo[i] * 8
                combo_array[0,i,2] = 6
        #combo_array = combo.reshape((1, len(combo)))
        self.selected_rank = array_image.show_array(
            combo_array, width=width, height=20, background="#888", hover_text_callback=None,
            widget=self.selected_rank, margin=10,
        )

    def perturb(self, errors):
        combo = self.combo.combo_array
        perturbation_array = ch.variations_of_max_size(combo, errors)
        print ("perturbation size", len(combo), perturbation_array.shape)
        ranker = RankOfRankingsViz(
            primary_stat=self.primary_stat,
            secondary_stat=self.secondary_stat,
            n=len(combo),
            k=1,
            max_n=self.max_n,
            width=self.width,
            combinations_array=perturbation_array
        )
        return ranker

    def rankings_hover(self, col, row, array):
        combo = self.rank.combination(col)
        s = combo_str(combo.combo_array)
        return "%s : %s=%4.2f; %s=%4.2f" % (s, combo.metric1.abbreviation, combo.r1, combo.metric2.abbreviation, combo.r2)

    def set_n_k(self, n, k):
        self.n = n
        self.k = k

def combo_str(combo):
    return "".join([str(c) for c in combo])

class curve_circle_callbacks:

    def __init__(self, circle, in_frame, threshold, viz, x, y, marker):
        (self.circle, self.in_frame, self.threshold) = (circle, in_frame, threshold)
        self.viz = viz
        self.marker = marker
        self.x = x
        self.y = y
        circle.on("click", self.click)
        circle.on("mouseover", self.click)

    def hover(self, event):
        #elf.viz.info.value = "mouseover" + repr((self.x,self.y))
        self.marker.change(x=self.x, y=self.y)

    def click(self, event):
        self.marker.change(x=self.x, y=self.y)
        self.viz.rebuild_combo_array(self.threshold)


def n_k_slider(update_callback, n=7, k=3, max_n=50, width=600, height=50):
    w = jp_proxy_widget.JSProxyWidget()
    w.check_jquery()
    w.js_init("""
        element.empty();
        element.width(width);
        element.height(height);
        var slider_div = $("<div/>").appendTo(element);
        //slider_div.height(height);
        slider_div.slider({
            range: true,
            min: 0,
            max: max_n,
            values: [k, n],
            slide: function(event, ui) {
                element.on_slide(event, ui);
            }
        });
        element.slider_div = slider_div
        element.info_div = $("<div/>").appendTo(element);
        element.on_slide = function(event, ui) {
            var k = ui.values[0];
            var n = ui.values[1];
            element.display_info(n, k);
            if (callback) {
                callback(n, k);
            }
        };
        element.display_info = function(n, k) {
            element.info_div.html("n=" + n + "; k=" +k);
        };
        element.display_info(n, k);
    """, n=n, k=k, max_n=max_n, callback=update_callback, width=width, height=height)
    return w

