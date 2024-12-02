classdef Color2VectorConverter < handle
    % A Color-to-Vector Converter class definition.
    %
    % Intended use: conversion of 2 sets of Doppler Color data 
    % into a single set of Doppler Vector data.
    
    properties(Access = private)
        
        powerDropLim
        ang0
        ang1
    end
    
    methods
        
        function obj = Color2VectorConverter(txAng0,txAng1,rxAng0,rxAng1,powerDropLim)
            % Creates Color2VectorConverter object
            %
            % :param txAng0: tx angle, 1st projection
            % :param txAng1: tx angle, 2nd projection
            % :param rxAng0: rx angle, 1st projection
            % :param rxAng1: rx angle, 2nd projection
            % :param powerDropLim: power difference limit [dB] between
            %   projections, above which the weaker projection is locally
            %   neglected
            % :returns: Color2VectorConverter object
            
            obj.powerDropLim = powerDropLim;
            
            obj.ang0 = (txAng0 + rxAng0)/2;
            obj.ang1 = (txAng1 + rxAng1)/2;
            
        end
        
        function [color,power,turbu] = convert(obj,color0,color1,power0,power1,turbu0,turbu1)
            % Performs Color-to-Vector conversion
            %
            % :param color0: color data, 1st projection
            % :param color1: color data, 2nd projection
            % :param power0: power data, 1st projection
            % :param power1: power data, 2nd projection
            % :param turbu0: turbulence data, 1st projection
            % :param turbu1: turbulence data, 2nd projection
            % :returns: converted color/power/turbulence data
            
            mask0 = (power0>0 & power1>0) & (10*log10(power0 ./ max(power0,power1)) >= -obj.powerDropLim);
            mask1 = (power0>0 & power1>0) & (10*log10(power1 ./ max(power0,power1)) >= -obj.powerDropLim);

            color = cat(4, (color0.*mask0./cos(obj.ang0) - ...
                            color1.*mask1./cos(obj.ang1))                / (tan(obj.ang1) - tan(obj.ang0)), ... % x-color
                           (color1.*mask1./cos(obj.ang1).*tan(obj.ang0) - ...
                            color0.*mask0./cos(obj.ang0).*tan(obj.ang1)) / (tan(obj.ang1) - tan(obj.ang0)) );   % z-color
            power = power0.*mask0 + power1.*mask1;
            turbu = max(turbu0,turbu1);
        end
        
    end
end


