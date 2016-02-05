Y = loadMNISTImages('t10k-images.idx3-ubyte');
Yla = loadMNISTLabels('t10k-labels.idx1-ubyte');
% numero de pixeles
ninputs = size(Y,1);
% numero de neuronas 1ª capa
nhidden = ninputs;
% numero de imagenes
numimagenes = size(Y,2);
% resultados para comparar luego
result = zeros(size(Yla));
dif = 0


%%
%Wih, Who, Bh y Bo
%deben estar calculados previamente
%%
for i=1:1:numimagenes
    %% Cálculos hacia delante
    % Valores de la capa oculta
    H = Y(:,i)'*Wih+Bh;
    H = ajusta(H,52.7,170);
%     Valores de la capa de salida
    O = H*Who+Bo;
    O = ajusta(O,196,207);
    %resultados de 0 a 9 discreto
    result(i) = O;
    aciertos = aciertos + ~(10*Yla(i)-round(10*O));
    dif = dif + (la(i)-O)^2;
    mse = dif/i;
end
dif = Yla-round(10*result);
% contamos los aciertos (diferencia nula)
suma = sum(~dif);
tasa = suma/numimagenes
% if tasa<0.3
%     disp('FATAAAAAAAAAAL')
% end