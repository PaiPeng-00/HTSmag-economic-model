function cp5_extract()
  addpath(fullfile(getenv('COMSOL_ROOT'),'mli')); mphstart(2036);
  import com.comsol.model.util.*;
  base=[fullfile(getenv('HTSMAG_FEM_DATA_ROOT'),'2-TA-loss') filesep];
  res=[getenv('HTSMAG_RESULTS_ROOT') filesep];
  try
    m=mphload([base 'ARC_TA_loss_solved.mph']); fprintf('LOADED solved\n');
    % compute each loss node into a fresh table, then read it
    magD = node_to_table(m,'int4'); fprintf('int4(mag) -> %s\n', sz(magD));
    radD = node_to_table(m,'gev3'); fprintf('gev3(radial) -> %s\n', sz(radD));
    disp('mag tail:'); disp(magD(max(1,end-2):end,:));
    disp('radial tail:'); disp(radD(max(1,end-2):end,:));
    % save raw tables
    writematrix(magD,[res 'ARC_magloss_Npw10_rho5000_T20.csv']);
    writematrix(radD,[res 'ARC_radialloss_Npw10_rho5000_T20.csv']);
    fprintf('WROTE CSVs\nEXTRACT_OK\n');
  catch ME
    fprintf(2,'EXTRACT_ERR: %s\n', getReport(ME,'extended','hyperlinks','off'));
  end
end
function D = node_to_table(m, tag)
  import com.comsol.model.util.*;
  tt = ['tmp_' tag];
  try; m.result.table.create(tt,'Table'); catch; end
  m.result.numerical(tag).set('table',tt);
  m.result.numerical(tag).set('data','dset1');
  m.result.numerical(tag).setResult;
  D = mphtable(m, tt).data;
end
function s = sz(D); s = sprintf('%dx%d', size(D,1), size(D,2)); end
