clear;
data = readData();
validation_set = cross_validation(data);

%part function
function out = readData()
    data = fopen('cross.pat');
    text_line = fgetl(data);
    out = zeros(1,4);
    index = 1;
    line = 1;
    while ischar(text_line)
        x = str2double(split(text_line));
        if mod(index,3)==2
            out(line,1) = x(1,1);
            out(line,2) = x(2,1);
        elseif mod(index,3)==0
            out(line,3) = x(1,1);
            out(line,4) = x(2,1);
            line = line+1;
        end
        index = index +1;
        text_line = fgetl(data);
    end
    fclose(data);
end

function data_train = cross_validation(x)
    x_rand = randperm(size(x,1));
    size_validation = fix(size(x,1)/10);
    data_train = zeros(size_validation,4,10);
    index = 1;
    in_i = 1;
    in_j = 1;
    for i = 1 : size(x,1)
        for j = 1 : size(x,2)
            data_train(in_i,in_j,index) = x(x_rand(i),j);
            in_j =in_j +1;
        end
        in_i = in_i +1;
        in_j = 1;
        if mod(i,size_validation) == 0
            index = index +1;
            in_i = 1;
        end
    end
end

function [w1,w2,w3,w4] = GenerateW(x,y,z,a)
    w_rand_range = -(1/(sqrt(numel(x)+length(y)+length(z)+1))):0.0001: (1/(sqrt(numel(x)+length(y)+length(z)+1)));
    w_rand_index = randperm(length(w_rand_range));
    
    for i = 1:size(x,1)
        for j = 1:size(x,2)
            x(i,j) = w_rand_range(w_rand_index(1));
            w_rand_index(1)=[];
        end
    end
    
    for i = 1:length(y)
        y(i) = w_rand_range(w_rand_index(1));
        w_rand_index(1)=[];
    end
    
    for i = 1:length(z)
        z(i) = w_rand_range(w_rand_index(1));
        w_rand_index(1)=[];
    end
    
    a = w_rand_range(w_rand_index(1));
    
    w1 = x;
    w2 = y;
    w3 = z;
    w4 = a;
end