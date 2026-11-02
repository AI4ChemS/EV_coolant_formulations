# Total plot width in pixel
s_widthpixel = 1000

# Border width scaling
s_borderscale = 1.0
# Grid width scaling
s_gridscale = 1.0
# Contour width scaling
s_contourscale = 1.0

# Title font scaling
s_titlefontscale = 1.0
# Axis title font scaling
s_labelfontscale = 1.0
# Tics font scaling
s_ticsfontscale = 1.0

# Aspect ratio
s_ratio = 1.000000

# Global title
s_title = ""
# X axis title
s_xtitle = "[H2r\\_O1o] Distance / pm"
# Y axis title
s_ytitle = "[O1r\\_H2r]-[H2r\\_O1o] Angle / Degree"

# X interpolation steps
s_xipl = 5
# Y interpolation steps
s_yipl = 5

# Draw contour lines (0 = No, 1 = Yes)
s_contour = 1
# Number of contour levels
s_ncontour = 30

# Plotting function
f_plot(x) = (x>s_zmax?s_zmax:x)**0.5

# Minimum X
s_xmin = 0.000000
# Maximum X
s_xmax = 1000.000000
# Major X tics distance
s_xtics = 200.000000
# Minor X tics number
s_mxtics = 2

# Minimum Y
s_ymin = 0.000000
# Maximum Y
s_ymax = 180.000000
# Major Y tics distance
s_ytics = 45.000000
# Minor Y tics number
s_mytics = 3

# Use automatic Z values (0 = No, 1 = Yes)
s_zauto = 1
# Minimum Z
s_zmin = 0
# Maximum Z
s_zmax = 14110.197313
# Major Z tics distance
s_ztics = 500
# Minor Z tics number
s_mztics = 1

# Legend orientation (0 = Horizontal, 1 = Vertical)
s_lorient = 0

# The red line
#set arrow 1 from 50,50 to 200,200 nohead back linewidth 0.7 linecolor rgbcolor "red"

# Do not change anything below

f_signpow(x,p) = sgn(x)*(abs(x)**p)

s_scale = s_widthpixel / 1024.0

set term unknown

splot "cdf_2_rdf[H2r_O1o]_adf[O1r_H2r]-[H2r_O1o].gp.csv" matrix using ($2*10+5):($1*1.8+0.9):3

if (s_zauto) s_zmin = GPVAL_Z_MIN; s_zmax = GPVAL_Z_MAX

font_string = sprintf("%s,%d", "Helvetica", 24*s_scale)
titlefont_string = sprintf("%s,%d", "Helvetica", 24*1.2*s_scale*s_titlefontscale)
labelfont_string = sprintf("%s,%d", "Helvetica", 24*s_scale*s_labelfontscale)
ticsfont_string = sprintf("%s,%d", "Helvetica", 24*s_scale*s_ticsfontscale)

set term pngcairo enhanced font font_string linewidth 4.0*s_scale size 1024*s_scale,1024*s_scale
set output "cdf_2_rdf[H2r_O1o]_adf[O1r_H2r]-[H2r_O1o].png"
set size ratio s_ratio

set title s_title font titlefont_string
set xlabel s_xtitle font labelfont_string
set ylabel s_ytitle font labelfont_string
unset key
unset clabel

set pm3d interpolate s_xipl,s_yipl map

set palette model RGB functions gray<=0.0 ? 1.0 : gray<0.2 ? 1.0-(5.0**1.5*gray**1.5)*0.8 : gray<0.4 ? 0.2-(gray-0.2) : gray<0.6 ? 5.0**0.5*(gray-0.4)**0.5 : gray<0.8 ? 1.0 : gray<1.0 ? 1.0 : 1.0, gray<=0.0 ? 1.0 : gray<0.2 ? 1.0-(5.0**1.5*gray**1.5)*0.8 : gray<0.4 ? 0.2+0.8*(5.0**0.75*(gray-0.2)**0.75) : gray<0.6 ? 1.0 : gray<0.8 ? 1.0-5.0*(gray-0.6) : gray<1.0 ? 0.0 : 0.0, gray<=0.0 ? 1.0 : gray<0.2 ? 1.0 : gray<0.4 ? 1.0-5.0**1.33*(gray-0.2)**1.33 : gray<0.6 ? 0.0 : gray<0.8 ? 0.0 : gray<1.0 ? 5.0*(gray-0.8) : 1.0

if (s_contour) set palette maxcolors s_ncontour; else set palette maxcolors 0

set xrange [s_xmin:s_xmax]
set yrange [s_ymin:s_ymax]
set zrange [f_plot(s_zmin):f_plot(s_zmax)]
set cbrange [f_plot(s_zmin):f_plot(s_zmax)]
unset colorbox

set tics font ticsfont_string
set xtics s_xtics out scale 0.5*s_borderscale offset 0,0.5
set mxtics s_mxtics
set ytics s_ytics out scale 0.5*s_borderscale offset 0.5,0
set mytics s_mytics

set border linewidth s_borderscale

set multiplot

set origin 0.0, 0.0
set size 1.1, 1.1

splot "cdf_2_rdf[H2r_O1o]_adf[O1r_H2r]-[H2r_O1o].gp.csv" matrix using ($2*10+5):($1*1.8+0.9):(f_plot($3))

set grid xtics mxtics ytics mytics back linestyle -1 linewidth 0.15*s_gridscale linecolor rgbcolor "gray40"
set title ""
set xlabel ""
set ylabel ""
set xtics in format ""
set ytics in format ""
if (s_contour) set contour base; set cntrparam bspline; set cntrparam levels incremental f_plot(s_zmin),1.0*(f_plot(s_zmax)-f_plot(s_zmin))/s_ncontour,f_plot(s_zmax)

splot "cdf_2_rdf[H2r_O1o]_adf[O1r_H2r]-[H2r_O1o].gp.csv" matrix using ($2*10+5):($1*1.8+0.9):(s_contour ? f_plot($3) : 1/0) with lines linewidth 0.15*s_contourscale linecolor rgbcolor "black" nosurface

unset multiplot

if (s_lorient) t_x = 196*s_scale; else t_x = 896*s_scale
if (s_lorient) t_y = 896*s_scale; else t_y = 196*s_scale

set term pngcairo enhanced font font_string linewidth 4.0*s_scale size t_x,t_y
set output "cdf_2_rdf[H2r_O1o]_adf[O1r_H2r]-[H2r_O1o]_box.png"

if (s_lorient) t_ratio = 20; else t_ratio = 0.05

set origin -0.13,-0.13
set size ratio t_ratio 1.25,1.25

unset title
unset xlabel
unset ylabel

set grid front
unset grid

if (s_lorient) set xrange [0:1]; else set xrange [s_zmin:s_zmax]
if (s_lorient) set yrange [s_zmin:s_zmax]; else set yrange [0:1]
if (s_lorient) set format y; else set format x
if (s_zauto) set xtics scale 0.5*s_borderscale offset s_lorient ? -0.5 : 0,s_lorient ? 0 : 0.5 autofreq; else set xtics scale 0.5*s_borderscale offset 0,0.5 s_ztics
if (s_zauto) set mxtics default; else set mxtics s_mztics
if (s_zauto) set ytics scale 0.5*s_borderscale offset s_lorient ? -0.5 : 0,s_lorient ? 0 : 0.5 autofreq; else set ytics scale 0.5*s_borderscale offset 0,0.5 s_ztics
if (s_zauto) set mytics default; else set mytics s_mztics
if (s_lorient) unset xtics; else unset ytics

if (s_lorient) set isosamples 2,500

if (s_lorient) set view 0,0; set border 15; unset ztics; set size 0.85,1.5; set origin -0.13,-0.25

splot f_plot(s_lorient ? y : x) nocontour, s_contour ? f_plot(s_lorient ? y : x) : 1/0 with lines linewidth 0.15*s_contourscale linecolor rgbcolor "black" nosurface
