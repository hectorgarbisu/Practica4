X = loadMNISTImages('train-images.idx3-ubyte');
la = loadMNISTLabels('train-labels.idx1-ubyte');
% etiquetas entre 0 y 1;
la = la/10;
% numero de pixeles
ninputs = size(X,1);
% numero de neuronas 1ª capa
nhidden = 200;
% numero de imagenes
numimagenes = size(X,2);
% neuronas de salida
noutput = 1;
% factor de aprendizaje
alph = 0.1;

% Pesos y Bias iniciales
% INPUT LAYER 785 neuron
Wih = rand(ninputs,nhidden);
% HIDDEN LAYER (?)
Who = rand(nhidden,noutput);
Bh = rand(1,nhidden);
% OUTPUT LAYER
Bo = rand(1,noutput);

% Variables auxiliares
incaciertos = zeros(numimagenes,1);
incmse = zeros(numimagenes,1);
aciertos = 0;
dif = 0;
mediaA = 51.7;
Hes = zeros(numimagenes,nhidden);
Oes = zeros(numimagenes,1);
%% Entrenamiento
tic
niteraciones = 10;
for j=1:1:niteraciones
for i=1:1:numimagenes
    %% Cálculos hacia delante
    % Valores de la capa oculta
    H = X(:,i)'*Wih+Bh;
    H = ajusta(H,52.7,170);
    Hes(i,:) = H;
%     Valores de la capa de salida
    O = H*Who+Bo;
    O = ajusta(O,196,207);
    Oes(i) = O;

    % Cálculos hacia atrás
%     Deltas
    dO = O*(1-O)*(la(i)-O);
    dH = H*(1-H')*dO*Who;
%     dO = (la(i)-O);
%     dH = dO*Who;

%     Reevaluación de Pesos y Bias
    Who = Who + alph.*H'.*dO;
    Bo = Bo + alph.*dO;
    Wih = Wih + alph.*X(:,i)*dH';
    Bh = Bh + alph.*dH';
%     
    aciertos = aciertos + ~(10*la(i)-round(10*O));
    incaciertos(i) = aciertos;
    dif = dif + (la(i)-O)^2;
    mse = dif/i;
    incmse(i) = mse;
    if(mod(i,6000)==0)
        disp(strcat('training: ',num2str(i),'/ ',num2str(numimagenes)));
        disp(strcat('aciertos :',num2str(aciertos),' ( ',num2str(aciertos/i),'/1)'));
        disp(strcat('mse :',num2str(mse)));
    end
end
end
plot(incmse)
[alph nhidden mse(end)]
toc
%% 10
%   10.0000   10.0000    0.0531 <-
%    3.0000   10.0000    0.0535
%    1.0000   10.0000    0.0563
%    0.2000   10.0000    0.0799
%% 30
%    3.0000   30.0000    0.0562
%    1.0000   30.0000    0.0511
%    0.3000   30.0000    0.0472 <-
%    0.1000   30.0000    0.0601
%% 100
%    0.3000  100.0000    0.0383
%    0.2000  100.0000    0.0372 <-
%    0.1500  100.0000    0.0373
%    0.1000  100.0000    0.0378
%    0.0300  100.0000    0.0467
%% 200
%    0.3000  200.0000    0.0370
%    0.1000  200.0000    0.0353 <--
%    0.0300  200.0000    0.0376
%% 400    
%    0.3000  400.0000    0.0380
%    0.1000  400.0000    0.0357 <-
%    0.0300  400.0000    0.0363



