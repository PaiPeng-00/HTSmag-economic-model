function cp5_extract_mag()
  addpath(fullfile(getenv('COMSOL_ROOT'),'mli')); mphstart(2036);
  import com.comsol.model.util.*;
  base=[fullfile(getenv('HTSMAG_FEM_DATA_ROOT'),'2-TA-loss') filesep];
  res=[getenv('HTSMAG_RESULTS_ROOT') filesep];
  try
    m=mphload([base 'ARC_TA_loss_solved.mph']); fprintf('LOADED\n');
    try; m.result.table.create('tmpmag','Table'); catch; end
    m.result.numerical('int4').set('table','tmpmag');
    m.result.numerical('int4').set('data','dset1');
    m.result.numerical('int4').setResult;
    D = mphtable(m,'tmpmag').data;
    fprintf('mag table size %dx%d\n', size(D,1), size(D,2));
    fprintf('first rows:\n'); disp(D(1:min(3,end),:));
    fprintf('last rows:\n');  disp(D(max(1,end-2):end,:));
    writematrix(D, [res 'ARC_magloss_Npw10_rho5000_T20.csv']);
    % col2 is mag loss [W]; report peak (single TF coil)
    if size(D,2)>=2
      fprintf('mag-loss: peak=%.3f W, end=%.3f W (per TF coil)\n', max(D(:,2)), D(end,2));
    end
    fprintf('EXTRACT_MAG_OK\n');
  catch ME
    fprintf(2,'ERR: %s\n', getReport(ME,'basic','hyperlinks','off'));
  end
end
