function [bestLabels, all_centers] = kmeans(im,centers,m,max_iter,S)
    % centers déjà initialisé
    all_centers = zeros([size(centers) max_iter]);
    k = size(centers,1); 
    it = 0;
    bestLabels = zeros(size(im,1),size(im,2),max_iter);
    while it < max_iter
        it = it + 1;
        sommes = zeros(k,6);
        for i=1:size(im,1)
            for j=1:size(im,2)
                c_proches = (centers(:,1) < (i + S)) .* (centers(:,1) > (i - S)) .* (centers(:,2) > (j - S)) .* (centers(:,2) < (j + S));
                mini = inf;
                ind = -1;
                for l=1:size(centers,1)
                    if c_proches(l) == 1
                        distance = sqrt(sum((double(centers(l,1:2)) - repmat(double([i j]),k, 1)).^2,2)) .* (m/S) + ...
                            sqrt(sum((double(centers(l,3:5)) - repmat(double(reshape(im(i,j,:),[1 3])),k, 1)).^2,2));
                        if distance < mini
                            mini = distance;
                            ind = l;
                        end
                    end
                end
                bestLabels(i,j,it) = ind;

                sommes(ind,1) = double(sommes(ind,1) + 1);
                sommes(ind,2:6) = double(sommes(ind,2:6)) + [double(i) double(j) double(reshape(im(i,j,:),[1 3]))];
            end
        end
        centers = sommes(:,2:6) ./ repmat(sommes(:,1),1,5);
        all_centers(:,:,it) = centers;
    end
end

