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
alph = 1;

%% Pesos y Bias iniciales
% HIDDEN LAYER (?)
Wih = rand(ninputs,nhidden);
Bh = rand(1,nhidden);
% OUTPUT LAYER
Who = rand(nhidden,noutput);
Bo = rand(1,noutput);
% Variables auxiliares
aciertos = 0;
dif = 0;
incaciertos = zeros(numimagenes,1);
incmse = zeros(numimagenes,1);
Hes = zeros(numimagenes,nhidden);
Oes = zeros(numimagenes,noutput);
%% Entrenamiento
tic
niteraciones = 10;
for j=1:1:niteraciones
aciertos = 0;
dif = 0;
incaciertos = zeros(numimagenes,1);
incmse = zeros(numimagenes,1);
Hes = zeros(numimagenes,nhidden);
Oes = zeros(numimagenes,noutput);
disp(strcat('epoca :',num2str(j),'/',num2str(niteraciones)));
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
    
    %% Calculo del coste
    distancia = lavec(i,:)-O;
    
    %% Cálculos hacia atrás APRENDIZAJE
%     Deltas
    dO  = O.*(1-O).*distancia;
    dH1 = H1.*(1-H1).*(dO*Who');
    
% %     Reevaluación de Pesos y Bias
    Who = Who + alph.*H1'*dO;
    Bo  = Bo  + alph.*dO;

    Wih = Wih + alph.*X(:,i)*dH1;
    Bh = Bh + alph.*dH1;
   
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
end
[alph nhidden noutput mse(end)]
plot(incmse)
toc
% [p,t] = max(Oes')
% [la t'-1]
%% funcionando :)
%    alpha   |H|        |O|        mse al final del entrenamiento

%%           200 
%    0.1000  200.0000   10.0000    0.1712
%    0.0300  200.0000   10.0000    0.0432 <-
%    0.0100  200.0000   10.0000    0.0436 <-
%    0.0100  200.0000   10.0000    0.0477 
%    0.0050  200.0000   10.0000    0.0528 
%    0.0030  200.0000   10.0000    0.0676 


%%           600
%    1.0000  600.0000   10.0000    0.0013 10 epocas %aciertos (0.9935/1)
%    1.0000  600.0000   10.0000    0.0162 <---- % NUEVA FORMULA YOOOLO
%    1.0000  600.0000   10.0000    0.0241
%    0.3000  600.0000   10.0000    0.0288
%    0.1000  600.0000   10.0000    0.0223 <- 
%    0.0100  600.0000   10.0000    0.0500
%    0.0010  600.0000   10.0000    0.0923
%
%    0.1000  600.0000   10.0000    0.0130 10 iteraciones
%


