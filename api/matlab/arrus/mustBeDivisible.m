function mustBeDivisible(a, b)
% Verifies if a is divisible by b.
    if mod(a, b) ~= 0
        error('ARRUS:IllegalArgument',...
             ['Value should be divisible by ', num2str(b)]);
    end
end

