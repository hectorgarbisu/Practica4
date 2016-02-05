X = loadMNISTImages('train-images.idx3-ubyte');
la = loadMNISTLabels('train-labels.idx1-ubyte');
% etiquetas entre 0 y 1;
la = la/10;
% numero de pixeles
ninputs = size(X,1);
% numero de neuronas 1ª capa
nhidden = 50;
% numero de imagenes
numimagenes = size(X,2);
% neuronas de salida
noutput = 1;
% factor de aprendizaje
alph = 0.00000001;

% Pesos y Bias iniciales
% INPUT LAYER 785 neuron
Wih = normrnd(0,1,ninputs,nhidden);
% HIDDEN LAYER (?)
Who = normrnd(0,1,nhidden,noutput);
Bh = normrnd(0,1,1,nhidden);
% OUTPUT LAYER
Bo = normrnd(0,1,1,noutput);

% Variables auxiliares
incaciertos = zeros(numimagenes,1);
incmse = zeros(numimagenes,1);
mse = 0;
aciertos = 0;
dif = 0;
Hes = zeros(numimagenes,nhidden);
Oes = zeros(numimagenes,1);
%% Entrenamiento
tic
for i=1:1:numimagenes
    %% Cálculos hacia delante
    % Valores de la capa oculta
    H = X(:,i)'*Wih+Bh;
%     H = ajusta(H,52.7,170);
    H = H/7.19; %divide entre std
    H = tanh(H); %tanh de una normal 0,1 (mola mas)
    H = (H+1)/2;
    Hes(i,:) = H;
%     Valores de la capa de salida
    O = H*Who+Bo;
    % resta media y divide por std para normal 0,1
%     O = (O-1.3)/2.1;
%     O = tanh(O);
% %     O = ajusta(O,196,207);
    O = (O+1)/2;
    Oes(i) = O;
% 
%     % Cálculos hacia atrás
% %     Deltas
%     dO = O*(1-O)*(la(i)-O);
%     dH = H*(1-H')*dO*Who;
% %     dO = (la(i)-O);
% %     dH = dO*Who;
% 
% % %     Reevaluación de Pesos y Bias
%     Who = Who + alph.*H'.*dO;
%     Bo = Bo + alph.*dO;
%     Wih = Wih + alph.*X(:,i)*dH';
%     Bh = Bh + alph.*dH';
% % %     
%     aciertos = aciertos + ~(10*la(i)-round(10*O));
%     incaciertos(i) = aciertos;
%     dif = dif + (la(i)-O)^2;
%     mse = dif/i;
%     incmse(i) = mse;
    if(mod(i,6000)==0)
        disp(strcat('training: ',num2str(i),'/ ',num2str(numimagenes)));
        disp(strcat('aciertos :',num2str(aciertos),' ( ',num2str(aciertos/i),'/1)'));
        disp(strcat('mse :',num2str(mse)));
    end
end
plot(incmse)
[alph nhidden mse(end)]
toc