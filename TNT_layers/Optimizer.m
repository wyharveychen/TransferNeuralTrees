classdef Optimizer < handle 
    properties
        
        G; % gradient^2 sum(adagrad)/ moving average(adadelta, RMSprop, Adam) 
        D; % x^2 moving average(adadelta)
        M; % momentum moving average (Adam)
        G_decay;
        D_decay;
        M_decay;
        
        learn_rate;        
    end

    methods 
        function optim = Optimizer()
            if(isempty(Optimizer.Method()))
                Optimizer.Method('RMSprop'); %Default optimizer
            end
            optim.SetLearnRate();
    
        end
        function SetLearnRate(optim)
            md = Optimizer.Method();                              
            if(strcmp(md,'Adam'))
                optim.learn_rate = 5e-2;
                optim.G = 0; optim.G_decay = 0.999;
                optim.M = 0; optim.M_decay = 0.9;                
            elseif(strcmp(md,'RMSprop'))
                optim.learn_rate = 5e-3;
                optim.G = 0; optim.G_decay = 0.9;
            elseif(strcmp(md,'adadelta'))
                optim.learn_rate = 5;
                optim.G = 0; optim.G_decay = 0.9;
                optim.D = 0; optim.D_decay = 0.9;
            elseif(strcmp(md,'adagrad'))
                optim.learn_rate = 5e-2;
                optim.G = 0;
            else
                optim.learn_rate = 1;
            end
        end
        
        function weights = WeightUpdate(optim,weights,grad,reg_lambda)
            if( ~exist('reg_lambda','var'))
                learn_rate = optim.learn_rate;
            else
                learn_rate = reg_lambda * optim.learn_rate;
            end
            md = Optimizer.Method();     
            if(iscell(grad))
                %Cell initialization problem is handled by Util.CellFunwConst
                %One need not to initalize zero-constant cell to perform
                %accumulation, but only need to initalize it as 0, since it
                %will add 0 to the gradient                
                                                                
                if(strcmp(md,'Adam'))
                    optim.G = Util.CellFunwConst(@(g,G,G_d)     (   G_d*G + (1-G_d)*g.^2),  grad,   optim.G,    optim.G_decay);
                    optim.M = Util.CellFunwConst(@(g,M,M_d)     (   M_d*M + (1-M_d)*g),     grad,   optim.M,    optim.M_decay);
                    weights = Util.CellFunwConst(@(w,g,G,M,G_d,M_d,lr)  (w - lr*(M/(1-M_d))./(sqrt(G/(1-G_d))+1e-8)), weights, grad, optim.G, optim.M, optm.G_decay, optim.M_decay, learn_rate);
                elseif(strcmp(md,'RMSprop'))         
                    optim.G = Util.CellFunwConst(@(g,G,G_d)     (   G_d*G + (1-G_d)*g.^2),  grad,   optim.G,    optim.G_decay);
                    weights = Util.CellFunwConst(@(w,g,G,lr)    (w - lr*g./(sqrt(G)+1e-8)), weights, grad, optim.G, learn_rate);                   
                elseif(strcmp(md,'adadelta'))          
                    optim.G = Util.CellFunwConst(@(g,G,G_d) (   G_d*G + (1-G_d)*g.^2),          grad,   optim.G,    optim.G_decay);
                    X       = Util.CellFunwConst(@(g,G,D)   (   sqrt(D+0.1)./sqrt(G+0.1).*g),   grad,   optim.G,    optim.D);
                    optim.D = Util.CellFunwConst(@(x,D,D_d) (   D_d*D + (1-D_d)*x.^2),          X,      optim.D,    optim.D_decay);
                    weights = Util.CellFunwConst(@(w,x,lr)  (   w - lr*x),                      weights,X,          learn_rate);
                elseif(strcmp(md,'adagrad'))
                    optim.G = Util.CellFunwConst(@(g,G) (G + g.^2), grad ,optim.G);
                    weights = Util.CellFunwConst(@(w,g,G,lr) (w - lr*g./(sqrt(G)+1e-8)), weights, grad, optim.G, learn_rate);
                else
                    weights = Util.CellFunwConst(@(w,g,lr) (w - lr*g), weights, grad, learn_rate);
                end
            else
                if(strcmp(md,'Adam'))                    
                    optim.G = optim.G_decay*optim.G + (1-optim.G_decay)*grad.^2;
                    optim.M = optim.M_decay*optim.M + (1-optim.M_decay)*grad;
                    weights = weights -  learn_rate *(optim.M/(1-optim.M_decay))./(sqrt(optim.G/(1-optim.G_decay))+1e-8);
                elseif(strcmp(md,'RMSprop'))
                    optim.G = optim.G_decay*optim.G + (1-optim.G_decay)*grad.^2;
                    weights = weights -  learn_rate * grad./(sqrt(optim.G)+1e-8);
                elseif(strcmp(md,'adadelta'))
                    optim.G = optim.G_decay*optim.G + (1-optim.G_decay)*grad.^2;
                    x = sqrt(optim.D+0.1)./sqrt(optim.G+0.1).*grad;
                    optim.D = optim.D_decay*optim.D + (1-optim.D_decay)*x.^2;
                    weights = weights - learn_rate*x;
                elseif(strcmp(md,'adagrad'))
                    optim.G = optim.G + grad.^2;
                    weights = weights - learn_rate * grad./(sqrt(optim.G)+1e-8);
                else
                    weights = weights - learn_rate * grad;
                end
            end            
        end
    end

    methods(Static)
        function out = Method(md)
            persistent static_md;
            if(nargin)
                static_md = md;
            end
            out = static_md;
        end
    end
   
end