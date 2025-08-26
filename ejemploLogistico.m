%% Ejemplo de un clasificador binario (Regresor logistico)
%% PROGRAMA PRINCIPAL
clc,clear,clf
% Cargamos los datos
T = readtable("datosIris.txt");
% Extraemos las caracteristicas de largo/ancho 
% petalo/sepalo

X = T{:,[1,3]};
X = [ones(length(X),1),X];
%y = T{:,5:7};

% Cramos las etiquetas de clase
especie = {'setosa'; 'versicolor';'virginica'};
especie = repelem(especie,50,1);
% Creamos el vector de salidas correctas
y=double(strcmp(especie,'setosa'));
% Determinamos numero de datos y caracteristicas
[m,n] = size(X);


%% PARAMETROS DEL CLASIFICADOR
% inicializamos el vector de parametros 
w = zeros(n,1);
% Definimos la tasa de aprendizaje
eta = 0.6;
% Número de epocas
numEpocas = 1000;
% Definimos el gradiente de la función de costo
nablaJ = zeros(n,1);
% Creamos la funcion de activacion 
sigma = @(x) 1./(1+exp(-x));
logLoss = zeros(numEpocas,1);
% Mostramos la curva de error
subplot(2,1,2)
error = semilogy(logLoss);
grid on
ylim([1e-2,1])
xlim([0,numEpocas])
xlabel('Numero de épocas')
ylabel('Funcion de costo ')
%% Parametros para visualizar entrenamiento 

x1min=min( X(:,2) );
x1max=max( X(:,2) ) ;
x2min=min( X(:,3) );
x2max=max( X(:,3) ); 
x1f=[x1min,x1max]; 
%% Creamos los vectores para el mapa de probabilidad.
% Numero de pixeles
numPixel = 100;
x1=linspace(x1min,x1max,numPixel);
x2=linspace(x2min,x2max,numPixel);
% Creamos las matrices para las coordenadas 
[X1,X2] = meshgrid(x1,x2);
% Creamos la matriz de caracteristicas del mapa de probabilidades
Xmp = [ones(numPixel^2,1),X1(:),X2(:)];
% Inicializamos el mapa de probabilidad.
mapaP = zeros(numPixel,numPixel);

%% Mostramos clases
subplot(2,1,1)
mapaPG = imagesc(x1,x2,mapaP);
set(gca,'YDir','normal');
hold on
gscatter( X(:,2), X(:,3), especie )
xlabel('X_1')
ylabel('X_2')
ylim([0,7])
frontera = plot(x1f,[0,0]);
hold off

%% Entrenamos el regresor logistico (Clafisicador binario)
for q=1:numEpocas
    for j=1:n
        nablaJ(j)=0;
        for i=1:m
            % Extraemos el i-esimo vector de caracteristicas
            xi = X(i,:)';
            % Extraemos la i-esima salida correcta
            yi = y(i);
            nablaJ(j)= nablaJ(j) + (sigma(w'*xi)-yi)*xi(j);
        end
        nablaJ =nablaJ/m;
    end
    % Aplicamos el gradiente descendente
    w = w-eta*nablaJ;
    % Hacemos las predicciones : 
    p = sigma(X*w);
    % Calculamos el error usando la funcion logitsica 
    logLoss(q) = 0;
    for i=1:m
        % Extraemos el i-esimo vector de predicciones
        pi = p(i);
        % Extraemos la i-esima salida correcta
        yi = y(i);
        % Aplicamos la funcion logistica
        logLoss(q) = logLoss(q) - (yi*log( pi ) + ... 
                        ( 1-yi )*log( 1-pi ) )/m;
    end
    %% Evualamos el mapa de probabilidad
    mapaP = sigma(Xmp*w);
    mapaP = reshape(mapaP, numPixel, numPixel);
    %% Visualizacion de la frontera de decision
    b = -w(1)/w(3);
    x2f= - w(2)/w(3)*x1f + b;
    % Actualizo la frontera de decisión
    frontera.YData = x2f;
    % Actualizo el error
    error.YData = logLoss; 
    % ACtualizamos el mapa de probabilidad
    mapaPG.CData=mapaP;
    pause(0.01)
end

%  clf
% plot(strcmp(especie,'setosa'))
% hold on
% plot(classLogistico(X',w),'dr')
% sum(classLogistico(X',w) & strcmp(especie,'setosa'))
% classLogistico = @(x,w) sigma(x'*w) > 0.5;