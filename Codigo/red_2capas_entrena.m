X = loadMNISTImages('train-images.idx3-ubyte');
la = loadMNISTLabels('train-labels.idx1-ubyte');
% etiquetas entre 0 y 1;
la = la/10;
% numero de pixeles
ninputs = size(X,1);
% numero de neuronas 1ª capa
nhidden = 200;
% numero de neuronas 2ª capa
nhidde2 = 20;
% numero de imagenes
numimagenes = size(X,2);
% neuronas de salida
noutput = 1;
% factor de aprendizaje
alph = 0.01;

%% Pesos y Bias iniciales
% INPUT LAYER 784 neuron
% HIDDEN LAYER (?)
Wih = rand(ninputs,nhidden);
Bh = rand(1,nhidden);
% HIDDEN LAYER 2(?)
Whh = rand(nhidden,nhidde2);
Bh2 = rand(1,nhidde2);
% OUTPUT LAYER
Who = rand(nhidde2,noutput);
Bo = rand(1,noutput);
% Variables auxiliares
aciertos = 0;
dif = 0;
incaciertos = zeros(numimagenes,1);
incmse = zeros(numimagenes,1);
Hes = zeros(numimagenes,nhidden);
H2es = zeros(numimagenes,nhidde2);
Oes = zeros(numimagenes,1);
%% Entrenamiento
tic
for i=1:1:numimagenes
    %% Cálculos hacia delante
    %% Valores de la 1ª capa oculta
    H1 = X(:,i)'*Wih+Bh;
    H1 = ajusta(H1,52.7,170);
%     Hes(i,:) = H1;
    
    %% Valores de la 2ª capa oculta
    H2 = H1*Whh+Bh2;
    H2 = ajusta(H2,190,402);
%     H2es(i,:) = H2;
    
    %%  Valores de la capa de salida
    O = H2*Who+Bo;
    O = ajusta(O,3,4.2);
%     Oes(i) = O;
    
    %% Cálculos hacia atrás APRENDIZAJE
%     Deltas
    dO  = O*(1-O)*(la(i)-O);
    dH2 = H2*(1-H2')*dO*Who;
    dH1 = H1*(1-H1')*dH2'*Whh';
    
%     Reevaluación de Pesos y Bias
    Who = Who + alph.*H2'*dO;
    Bo  = Bo  + alph.*dO;
    
    Whh = Whh + alph.*H1'*dH2';
    Bh2 = Bh2 + alph.*dH2';
    
    Wih = Wih + alph.*X(:,i)*dH1;
    Bh = Bh + alph.*dH1;
   
    %% Comprobacion
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
[alph nhidden nhidde2 mse(end)]
toc
%%  100 10
%    0.1000  100.0000   10.0000    0.0403
%    0.0500  100.0000   10.0000    0.0403
%    0.0300  100.0000   10.0000    0.0382
%    0.0200  100.0000   10.0000    0.0376 <-
%    0.0100  100.0000   10.0000    0.0414

%% 200 10
%    0.3000  200.0000   10.0000    0.0442
%    0.1000  200.0000   10.0000    0.0410
%    0.0300  200.0000   10.0000    0.0373 <-
%    0.0200  200.0000   10.0000    0.0375
%    0.0100  200.0000   10.0000    0.0375
%
%% 200 20
%    0.1000  200.0000   20.0000    0.0426
%    0.0200  200.0000   20.0000    0.0383
%    0.0150  200.0000   20.0000    0.0388
%    0.0100  200.0000   20.0000    0.0356 <-
%    0.0300  200.0000   20.0000    0.0389

%% 200 50
%    0.1000  200.0000   50.0000    0.0838
%    0.0300  200.0000   50.0000    0.0419
%    0.0100  200.0000   50.0000    0.0388
%    0.0030  200.0000   50.0000    0.0359 <-
%    0.0010  200.0000   50.0000    0.0401

%% 200 100
%    0.1000  200.0000  100.0000    0.0841
%    0.0300  200.0000  100.0000    0.0617 <-
%    0.0100  200.0000  100.0000    0.0634
%    0.0010  200.0000  100.0000    0.0691

%% 400 40
%    0.0100  400.0000   40.0000    0.0675
%    0.0030  400.0000   40.0000    0.0544
%    0.0010  400.0000   40.0000    0.0469 
%    0.0003  400.0000   40.0000    0.0401
%    0.0001  400.0000   40.0000    0.0381 <-
%    0.00003 400.0000   40.0000    0.0400

%% 400 80
%    0.3000  400.0000   80.0000    0.0851
%    0.1000  400.0000   80.0000    0.0841
%    0.0300  400.0000   80.0000    0.0738
%    0.0100  400.0000   80.0000    0.0696 
%    0.0030  400.0000   80.0000    0.0633
%    0.0010  400.0000   80.0000    0.0518
%    0.0003  400.0000   80.0000    0.0484
%    0.0001  400.0000   80.0000    0.0450

%    0.0100  784.0000   10.0000    0.0376




