function lock_ip400()
  addpath(fullfile(getenv('COMSOL_ROOT'),'mli')); mphstart(2036);
  import com.comsol.model.util.*;
  base=[fullfile(getenv('HTSMAG_FEM_DATA_ROOT'),'1-field-FEM') filesep];
  try
    m=mphload([base 'ARC_field_mangiarotti.mph']);
    m.param.set('Ip','400[A]'); m.param.set('Nt','1750'); m.param.set('thk','0.3657[mm]');
    fprintf('SET Ip=400, Nt=1750, dr_TF=%.3f m, amp_turns=%.2f MA\n', ...
            m.param.evaluate('Nt')*m.param.evaluate('thk'), m.param.evaluate('amp_turns')/1e6);
    m.geom('geom1').run; m.mesh('mesh1').run;
    ts=tic; m.sol('sol1').runAll; fprintf('SOLVED %.0fs\n',toc(ts));
    R0cm = m.param.evaluate('R0')*100;
    B0   = mphinterp(m,'-mf.Bz','coord',[R0cm;0;0],'dataset','dset1')*400;
    Bmax = mphmax(m,'mf.normB','volume','selection',2,'dataset','dset1')*400;
    lf   = mphmax(m,'Ip/(Jc_mang*wid/10[mm])','volume','selection',2,'dataset','dset1');
    fprintf('=== Ip=400A 锁定确认 ===\n');
    fprintf('B0=%.2f T (9.2), Bmax=%.1f T (23), 载流比=%.3f\n', B0, Bmax, lf);
    mphsave(m,[base 'ARC_field_mangiarotti.mph']);
    fprintf('SAVED. LOCK400_OK\n');
  catch ME
    fprintf(2,'ERR: %s\n', getReport(ME,'basic','hyperlinks','off'));
  end
end
