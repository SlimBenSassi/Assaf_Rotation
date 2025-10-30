function zerolines()

h=gca;
line([h.XLim],[0 0],'LineStyle', '--', 'Color', [0 0 0])
line([0 0],[h.YLim],'LineStyle', '--', 'Color', [0 0 0])