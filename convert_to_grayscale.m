function grayscale_image_path = convert_to_grayscale(image_path)
    img = imread(image_path);
    gray_img = rgb2gray(img);
    [~, name, ext] = fileparts(image_path);
    grayscale_image_path = fullfile(tempdir, [name '_gray' ext]);
    imwrite(gray_img, grayscale_image_path);
end
