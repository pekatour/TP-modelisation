close all;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% CONSEIL : A METTRE DANS UN AUTRE SCRIPT %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
load donnees;

% Calcul des faces du maillage à garder
Nf = 4 * size(tri,1);
FACES = zeros(Nf,3);
combinaisons = boolean([1 1 1 0; 1 0 1 1; 1 1 0 1; 0 1 1 1]);
for k = 1:4:Nf
    copie = tri(ceil(k/4),:);
    for i=1:4
        FACES(k+i-1,:) = copie(combinaisons(i,:));
    end
end

% Tri des faces
for k = 1:Nf
    FACES(k,:) = sort(FACES(k,:));
end
FACES = sortrows(FACES);

% Sélection des faces externes
garder = boolean(ones(size(FACES,1),1));
for k = 2:Nf
    if isequal(FACES(k,:), FACES(k-1,:))
        garder(k-1:k,1) = 0;
    end
end
FACES = FACES(garder,:);


fprintf('Calcul du maillage final termine : %d faces. \n',size(FACES,1));

% Affichage du maillage final
figure;
hold on
for i = 1:size(FACES,1)
   plot3([X(1,FACES(i,1)) X(1,FACES(i,2))],[X(2,FACES(i,1)) X(2,FACES(i,2))],[X(3,FACES(i,1)) X(3,FACES(i,2))],'r');
   plot3([X(1,FACES(i,1)) X(1,FACES(i,3))],[X(2,FACES(i,1)) X(2,FACES(i,3))],[X(3,FACES(i,1)) X(3,FACES(i,3))],'r');
   plot3([X(1,FACES(i,3)) X(1,FACES(i,2))],[X(2,FACES(i,3)) X(2,FACES(i,2))],[X(3,FACES(i,3)) X(3,FACES(i,2))],'r');
end;