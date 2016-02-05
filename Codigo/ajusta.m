function [distribucion] = ajusta(distribucion,media,maximo)
    if (nargin<2)
        media = mean(distribucion);
        maximo = max(distribucion);
    end
    %centrada en 0
    distribucion = distribucion - media;
    %entre -3 y 3
    if (maximo~=0)
        distribucion = 3*distribucion/maximo;
    end
    %deslinealiza
    distribucion = tanh(distribucion);
    %entre 0 y 1
    distribucion = (distribucion+1)/2;
    
    
%     distribucion = 1./(1+(exp(-1*media*distribucion)));
end