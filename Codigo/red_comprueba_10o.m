X = loadMNISTImages('train-images.idx3-ubyte');
la = loadMNISTLabels('train-labels.idx1-ubyte');
% matriz de hot-zeros 
% La salida esperada es un 1 en la 
% neurona correspondiente
lavec = zeros(size(la,1),10);
for i=1:size(lavec,1)
    lavec(i,la(i)+1) = 1;
end
% numero de pixeles
ninputs = size(X,1);
% numero de neuronas 1ª capa
nhidden = 600;
% numero de imagenes
numimagenes = size(X,2);
% neuronas de salida
noutput = 10;
% factor de aprendizaje
% alph = 0.01;

%% Pesos y Bias iniciales
% ESTA PARTE DEBE VENIR DE red_entrena
% HIDDEN LAYER (?)
% Wih = rand(ninputs,nhidden);
% Bh = rand(1,nhidden);
% % OUTPUT LAYER
% Who = rand(nhidden,noutput);
% Bo = rand(1,noutput);
% Variables auxiliares
aciertos = 0;
dif = 0;
incaciertos = zeros(numimagenes,1);
incmse = zeros(numimagenes,1);
Hes = zeros(numimagenes,nhidden);
Oes = zeros(numimagenes,noutput);
%% Entrenamiento
tic
for i=1:1:numimagenes
    %% Cálculos hacia delante
    %% Valores de la 1ª capa oculta
    H1 = X(:,i)'*Wih+Bh;
    H1 = ajusta(H1,52.7,174);
    Hes(i,:) = H1;
    
    %%  Valores de la capa de salida
    O = H1*Who+Bo;
    O = ajusta(O,49,103);
    Oes(i,:) = O;
    

    %% Comprobacion
    [x,ind] = max(abs(O));
    aciertos = aciertos + lavec(i,ind);
    incaciertos(i) = aciertos;
    dif = dif + (((ind-1)-la(i,:))/10).^2;
    mse = dif/i;
    incmse(i) = mse;
    
    if(mod(i,6000)==0)
        disp(strcat('training: ',num2str(i),'/ ',num2str(numimagenes)));
        disp(strcat('aciertos :',num2str(aciertos),' ( ',num2str(aciertos/i),'/1)'));
        disp(strcat('mse :',num2str(mse)));
    end
end
[alph nhidden noutput mse(end)]
% plot(incmse)
toc
% [p,t] = max(Oes')
% [la t'-1]
%    0.0100  200.0000   10.0000    0.0242
%    0.1000  600.0000   10.0000    0.0128 10
%
%
%
%



