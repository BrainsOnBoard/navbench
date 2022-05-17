dat = load('test_unwrapped_20210303_153749');
test_image = dat.image(10,:,:)
imshow(shiftdim(test_image))

% Getting best-matching snap
train_dat = load('train')
snap = train_dat.image(dat.best_snap_idx(1)==train_dat.database_idx,:,:);
