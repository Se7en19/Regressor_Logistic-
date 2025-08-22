%% Ejemplo de un clasificador binario (Regresor logistico)
%% PROGRAMA PRINCIPAL
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
eta = 1;
% Número de epocas
numEpocas = 200;
% Definimos el gradiente de la función de costo
nablaJ = zeros(n,1);
% Creamos la funcion de activacion 
sigma = @(x) 1./(1+exp(-x));
%% Parametros para visualizar entrenamiento 
x1=[min( X(:,2) ),max( X(:,2) ) ]; 
gscatter( X(:,2), X(:,3), especie  )
xlabel('X_1')
ylabel('X_2')
hold on 
frontera = plot(x1,[0,0]);
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
    end
    % Aplicamos el gradiente descendente
    w = w-eta*nablaJ/m;
    %% Visualizacion de la frontera de decision
    b = -w(1)/w(3);
    x2 = - w(2)/w(3)*x1 + b;
    % Actualizo la frontera de decisión
    frontea.YData = x2;
    pause(0.01)
end
