function interpColorMap = InterpColorMap(cRgb,xq)
%%
assert(all(xq >= 0 & xq <= 1), ...
    'Interpolated points must be between 0 and 1')

if size(xq,1) > 1
    xq = xq';
end

cHsv = rgb2hsv(cRgb);

hInterp = interp1(linspace(0,1,size(cHsv,1)), cHsv(:,1), xq);
sInterp = interp1(linspace(0,1,size(cHsv,1)), cHsv(:,2), xq);
vInterp = interp1(linspace(0,1,size(cHsv,1)), cHsv(:,3), xq);
cHsv = [hInterp' sInterp' vInterp'];

interpColorMap = hsv2rgb(cHsv);


end