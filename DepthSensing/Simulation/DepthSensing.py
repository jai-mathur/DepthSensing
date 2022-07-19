
from cc3d import CompuCellSetup
        
from DepthSensingSteppables import UpdatePhi
CompuCellSetup.register_steppable(steppable=UpdatePhi(frequency=1))

from DepthSensingSteppables import DoRemodeling
CompuCellSetup.register_steppable(steppable=DoRemodeling(frequency=1))

from DepthSensingSteppables import FindECMpolarity
CompuCellSetup.register_steppable(steppable=FindECMpolarity(frequency=200))

from DepthSensingSteppables import SetCellForces
CompuCellSetup.register_steppable(steppable=SetCellForces(frequency=1))


CompuCellSetup.run()
 