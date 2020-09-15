angle = -30*pi/180;
probe = struct;
    probe.nElem = 32;
    probe.pitch = 0.3048;
c = 1490;


del = zeros(1,probe.nElem);

for i=1:probe.nElem
    if angle>=0
        n = 1;
    else
        n = probe.nElem;
    end
    del(i) = (i-n)*probe.pitch*tan(angle)/c;
end

plot(del)
